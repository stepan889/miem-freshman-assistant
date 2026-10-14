import asyncio
import html
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import (BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,
                           LinkPreviewOptions, Message, ReplyKeyboardMarkup, KeyboardButton)

from .agent_api import AgentAPI, AgentAPIError
from .assistant import Answer, Assistant
from .config import Settings
from .retrieval import KnowledgeBase, split_text

WELCOME = ("🎓 Помощник первокурсника ИВТ МИЭМ\n\n"
           "Отвечаю по программам 1 курса за 2026/2027: темы занятий, формулы оценивания, "
           "литература и связи дисциплин. К ответам прикладываю ссылки на ВШЭ.\n\n"
           "Напишите вопрос своими словами или выберите дисциплину. "
           "Укажите вариант программы, если он есть: обычный, углублённый или индустриальный.\n\n"
           "Вопрос и последние сообщения диалога передаются в Agent API для подготовки ответа.")
HELP = ("Примеры вопросов:\n"
        "• Как считается оценка по обычной алгебре и геометрии?\n"
        "• Какие книги рекомендуют по математическому анализу?\n"
        "• Какие темы изучают в обычном курсе информатики?\n"
        "• Как связаны обычные курсы алгебры и дискретной математики?\n\n"
        "/courses — выбрать дисциплину\n"
        "/all — вопросы по всем дисциплинам\n"
        "/sources — источники последнего ответа\n"
        "/reset — сбросить историю и выбор\n"
        "/help — примеры и команды\n\n"
        "Я использую публичные программы ВШЭ; материалы из закрытой LMS в базу не входят. "
        "Если сведений недостаточно, попрошу уточнение или сообщу об этом.")
MENU = ReplyKeyboardMarkup(keyboard=[
    [KeyboardButton(text="📚 Дисциплины"), KeyboardButton(text="💬 Примеры вопросов")],
    [KeyboardButton(text="🔗 Источники"), KeyboardButton(text="🔄 Сбросить контекст")],
], resize_keyboard=True, input_field_placeholder="Вопрос о дисциплине первого курса…")


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)
    selected: str | None = None
    last_answer: Answer | None = None
    last_request: float = 0
    touched: float = field(default_factory=time.monotonic)
    busy: bool = False
    revision: int = 0

    def reset(self):
        self.history.clear()
        self.selected = None
        self.last_answer = None
        self.revision += 1


class Sessions:
    def __init__(self, max_sessions=512, ttl=3600):
        self.items = OrderedDict()
        self.max_sessions, self.ttl = max_sessions, ttl

    def get(self, user_id: int):
        now = time.monotonic()
        for key in list(self.items):
            s = self.items[key]
            if not s.busy and now - s.touched > self.ttl:
                del self.items[key]
        if user_id not in self.items:
            if len(self.items) >= self.max_sessions:
                key = next((k for k, s in self.items.items() if not s.busy), None)
                if key is None:
                    raise RuntimeError("Все сессии заняты.")
                del self.items[key]
            self.items[user_id] = Session()
        s = self.items[user_id]
        s.touched = now
        self.items.move_to_end(user_id)
        return s


def course_keyboard(kb: KnowledgeBase, page=0):
    size = 8
    page = max(0, min(page, (len(kb.courses) - 1) // size))
    rows = [[InlineKeyboardButton(text=c["title"], callback_data=f"course:{c['id']}")]
            for c in kb.courses[page * size:(page + 1) * size]]
    nav = []
    if page:
        nav.append(InlineKeyboardButton(text="←", callback_data=f"page:{page-1}"))
    if (page + 1) * size < len(kb.courses):
        nav.append(InlineKeyboardButton(text="→", callback_data=f"page:{page+1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton(text="Все дисциплины", callback_data="all")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def send_sources(message: Message, answer: Answer):
    if not answer.sources:
        await message.answer("Для последнего ответа источники не использовались. Задайте вопрос о дисциплине.")
        return
    entries = [f'[{i}] <a href="{html.escape(c.url, quote=True)}">{html.escape(c.course)}</a>\n'
               f'{html.escape(c.section)}' for i, c in enumerate(answer.sources, 1)]
    part = "<b>Источники · ВШЭ, 2026/2027</b>\n\n"
    for entry in entries:
        if len(part) + len(entry) > 3000:
            await message.answer(part, parse_mode="HTML", link_preview_options=LinkPreviewOptions(is_disabled=True))
            part = ""
        part += entry + "\n\n"
    if part.strip():
        await message.answer(part, parse_mode="HTML", link_preview_options=LinkPreviewOptions(is_disabled=True))


async def send_answer(message, answer):
    for text in split_text(answer.text, size=1600):
        await message.answer(text, parse_mode=None, link_preview_options=LinkPreviewOptions(is_disabled=True))
    if answer.sources:
        await send_sources(message, answer)


async def typing_loop(bot: Bot, chat_id: int):
    while True:
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            # Потеря typing-индикатора не должна прерывать ответ.
            pass
        await asyncio.sleep(4)


def make_router(assistant: Assistant, settings: Settings):
    router = Router()
    router.message.filter(F.chat.type == "private")
    router.callback_query.filter(F.message.chat.type == "private")
    sessions = Sessions()
    semaphore = asyncio.Semaphore(settings.concurrency)
    kb = assistant.kb

    async def authorized(event):
        if settings.allowed_users and event.from_user.id not in settings.allowed_users:
            if isinstance(event, CallbackQuery):
                await event.answer("Доступ к этому прототипу ограничен.", show_alert=True)
            else:
                await event.answer("Доступ к этому прототипу ограничен владельцем.")
            return False
        return True

    @router.message(CommandStart())
    async def start(message: Message):
        if not await authorized(message):
            return
        sessions.get(message.from_user.id).reset()
        await message.answer(WELCOME, reply_markup=MENU)

    @router.message(Command("help"))
    @router.message(F.text == "💬 Примеры вопросов")
    async def help_message(message: Message):
        if await authorized(message):
            await message.answer(HELP, reply_markup=MENU)

    @router.message(Command("courses"))
    @router.message(F.text == "📚 Дисциплины")
    async def courses(message: Message):
        if await authorized(message):
            await message.answer(f"Выберите дисциплину · 1 курс, 2026/2027\nВсего: {len(kb.courses)}",
                                 reply_markup=course_keyboard(kb))

    @router.message(Command("all", "reset"))
    @router.message(F.text == "🔄 Сбросить контекст")
    async def reset(message: Message):
        if await authorized(message):
            sessions.get(message.from_user.id).reset()
            await message.answer("Контекст сброшен. Можно задать вопрос по любой дисциплине или сравнить несколько.")

    @router.message(Command("sources"))
    @router.message(F.text == "🔗 Источники")
    async def sources(message: Message):
        if await authorized(message):
            answer = sessions.get(message.from_user.id).last_answer
            if answer:
                await send_sources(message, answer)
            else:
                await message.answer("Пока нет ответа с источниками. Напишите вопрос о дисциплине.")

    @router.callback_query(F.data.startswith("page:"))
    async def page(callback: CallbackQuery):
        if not await authorized(callback):
            return
        await callback.answer()
        try:
            page_number = int(callback.data.split(":", 1)[1])
        except ValueError:
            return
        await callback.message.edit_reply_markup(reply_markup=course_keyboard(kb, page_number))

    @router.callback_query(F.data == "all")
    async def all_courses(callback: CallbackQuery):
        if not await authorized(callback):
            return
        sessions.get(callback.from_user.id).reset()
        await callback.answer()
        await callback.message.answer("Выбраны все дисциплины. Напишите вопрос, указав нужные предметы.")

    @router.callback_query(F.data.startswith("course:"))
    async def choose_course(callback: CallbackQuery):
        if not await authorized(callback):
            return
        course_id = callback.data.split(":", 1)[1]
        if course_id not in kb.by_id:
            await callback.answer("Дисциплина не найдена. Откройте /courses заново.", show_alert=True)
            return
        state = sessions.get(callback.from_user.id)
        state.reset()
        state.selected = course_id
        c = kb.by_id[course_id]
        await callback.answer()
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Темы", callback_data="example:topics"),
             InlineKeyboardButton(text="Оценивание", callback_data="example:grade"),
             InlineKeyboardButton(text="Литература", callback_data="example:books")],
            [InlineKeyboardButton(text="Сравнить с другой дисциплиной", callback_data="all")],
        ])
        await callback.message.answer(
            f"<b>{html.escape(c['title'])}</b>\n{html.escape(c['when'])} · 2026/2027\n\n"
            f"Напишите любой вопрос об этой дисциплине или выберите тему. Для сравнения используйте /all.\n\n"
            f'<a href="{html.escape(c["url"], quote=True)}">Программа на сайте ВШЭ</a>',
            parse_mode="HTML", reply_markup=keyboard, link_preview_options=LinkPreviewOptions(is_disabled=True))

    async def respond(message: Message, user_id: int, question: str):
        state = sessions.get(user_id)
        if state.busy:
            await message.answer("Ещё готовлю предыдущий ответ. Дождитесь его, пожалуйста.")
            return
        if time.monotonic() - state.last_request < settings.cooldown:
            await message.answer("Подождите несколько секунд перед следующим вопросом.")
            return
        if semaphore.locked():
            await message.answer("Сейчас готовлю несколько ответов. Попробуйте через несколько секунд.")
            return
        state.busy, state.last_request = True, time.monotonic()
        revision = state.revision
        typing = asyncio.create_task(typing_loop(message.bot, message.chat.id))
        try:
            async with semaphore:
                answer = await assistant.ask(question, list(state.history), state.selected)
            if state.revision != revision:
                await message.answer("Контекст изменился во время подготовки ответа. Задайте вопрос заново.")
                return
            await send_answer(message, answer)
            state.last_answer = answer
            state.history.extend([{"role": "user", "content": question[:4000]},
                                  {"role": "assistant", "content": answer.text[:2000]}])
            state.history = state.history[-4:]
        except AgentAPIError as e:
            await message.answer(str(e))
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка обработки запроса: %s", type(e).__name__)
            await message.answer("Не удалось подготовить ответ. Попробуйте ещё раз или уточните дисциплину.")
        finally:
            state.busy = False
            typing.cancel()
            await asyncio.gather(typing, return_exceptions=True)

    @router.callback_query(F.data.startswith("example:"))
    async def example(callback: CallbackQuery):
        if not await authorized(callback):
            return
        await callback.answer()
        questions = {"topics": "Какие темы изучают в этой дисциплине?",
                     "grade": "Как считается оценка по этой дисциплине? Укажи формулы по модулям.",
                     "books": "Какая основная и дополнительная литература рекомендована?"}
        question = questions.get(callback.data.split(":", 1)[1])
        if question:
            await respond(callback.message, callback.from_user.id, question)

    @router.message(F.text.startswith("/"))
    async def unknown_command(message: Message):
        if await authorized(message):
            await message.answer("Неизвестная команда. Используйте /help или напишите вопрос обычным сообщением.")

    @router.message(F.text)
    async def question(message: Message):
        if await authorized(message):
            await respond(message, message.from_user.id, message.text)

    @router.message()
    async def unsupported(message: Message):
        if await authorized(message):
            await message.answer("Пока принимаю только текстовые вопросы. Напишите вопрос сообщением.")

    return router


async def run_bot(settings: Settings):
    settings.require_keys(telegram=True)
    kb = KnowledgeBase()
    api = AgentAPI(settings)
    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()
    dp.include_router(make_router(Assistant(kb, api), settings))
    try:
        await bot.set_my_commands([BotCommand(command=k, description=v) for k, v in [
            ("start", "Начать"), ("courses", "Дисциплины первого курса"), ("all", "Все дисциплины"),
            ("sources", "Источники ответа"), ("reset", "Сбросить контекст"), ("help", "Примеры вопросов")]])
        # Поллинг не требует публичного сервера. Несколько копий с одним токеном не запускайте.
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await api.close()
        await bot.session.close()
