from datetime import datetime, timezone

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, Update, User
import pytest

from miem_bot.assistant import Answer
from miem_bot.config import Settings
from miem_bot.retrieval import KnowledgeBase
from miem_bot.telegram import Sessions, course_keyboard, make_router


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.methods = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.methods.append(method)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


class TestAssistant:
    __test__ = False

    def __init__(self):
        self.kb = KnowledgeBase()
        self.calls = []

    async def ask(self, question, history, selected):
        self.calls.append((question, history, selected))
        return Answer("Тестовый ответ [1].", [self.kb.chunks[0]])


def update(text, user_id=7, number=1):
    return Update(update_id=number, message=Message(message_id=number, date=datetime.now(timezone.utc),
                  chat=Chat(id=user_id, type="private"), from_user=User(id=user_id, is_bot=False, first_name="Студент"), text=text))


async def test_telegram_start_question_sources_and_reset():
    session = RecordingSession()
    bot = Bot("123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk", session=session)
    assistant = TestAssistant()
    dp = Dispatcher()
    dp.include_router(make_router(assistant, Settings(cooldown=0)))
    try:
        for i, text in enumerate(["/start", "Как считается оценка?", "/sources", "/reset", "А литература?"], 1):
            await dp.feed_update(bot, update(text, number=i))
        assert len(assistant.calls) == 2
        assert assistant.calls[1][1] == []  # /reset очистил диалог.
        messages = [m.text for m in session.methods if hasattr(m, "text")]
        assert any("Источники" in text and "hse.ru" in text for text in messages)
        assert any("Контекст сброшен" in text for text in messages)
    finally:
        await bot.session.close()


async def test_allowlist_blocks_model_requests():
    session = RecordingSession()
    bot = Bot("123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk", session=session)
    assistant = TestAssistant()
    dp = Dispatcher()
    dp.include_router(make_router(assistant, Settings(allowed_users={99})))
    await dp.feed_update(bot, update("Привет", user_id=7))
    assert not assistant.calls
    assert "ограничен" in session.methods[-1].text
    await bot.session.close()


def test_session_cache_is_bounded_and_isolated():
    sessions = Sessions(max_sessions=2)
    sessions.get(1).selected = "one"
    assert sessions.get(2).selected is None
    sessions.get(3)
    assert len(sessions.items) == 2 and 1 not in sessions.items


def test_all_course_pages_reachable():
    kb = KnowledgeBase()
    shown = []
    for page in range(4):
        keyboard = course_keyboard(kb, page)
        shown.extend(b.callback_data.split(":", 1)[1] for row in keyboard.inline_keyboard
                     for b in row if b.callback_data.startswith("course:"))
    assert set(shown) == set(kb.by_id)
