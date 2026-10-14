import argparse
import asyncio
import logging

import httpx

from .agent_api import AgentAPI, AgentAPIError
from .assistant import Assistant
from .config import Settings
from .ingest import ingest
from .retrieval import KnowledgeBase


async def execute(args):
    if args.command == "ingest":
        snapshot = await ingest()
        print(f"Сохранено {len(snapshot['courses'])} дисциплин; страниц каталога: {len(snapshot['catalog_pages'])}.")
        return
    settings = Settings.load()
    if args.command == "check-api":
        # Для списка моделей ключ не нужен; поиск по поставщику, затем точное сравнение.
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get("https://api.agentapi.ru/models", params={"q": "deepseek", "limit": 100})
            response.raise_for_status()
            ids = {m["id"] for m in response.json()["data"]}
        if settings.model not in ids:
            raise ValueError(f"Модель {settings.model} не найдена в публичном каталоге Agent API.")
        print(f"Модель доступна в каталоге: {settings.model}")
        if settings.api_key:
            api = AgentAPI(settings)
            try:
                result = await api.complete_json('Верни JSON {"ok":true}.', {"test": True}, max_tokens=64)
                if result.get("ok") is not True:
                    raise ValueError("Тестовый запрос не вернул ожидаемый ответ.")
                print("Ключ и генерация проверены.")
            finally:
                await api.close()
        else:
            print("AGENTAPI_KEY не заполнен: проверен каталог, генерация не проверена.")
        return
    kb = KnowledgeBase()
    if args.command == "search":
        for c in kb.search(args.question, limit=6):
            print(f"\n{c.course} — {c.section}\n{c.text}\n{c.url}")
        return
    if args.command == "ask":
        settings.require_keys()
        api = AgentAPI(settings)
        try:
            answer = await Assistant(kb, api).ask(args.question)
            print(answer.plain())
        finally:
            await api.close()
        return
    from .telegram import run_bot
    await run_bot(settings)


def main():
    parser = argparse.ArgumentParser(description="Telegram-помощник первокурсника ИВТ МИЭМ")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Запустить Telegram-бота")
    sub.add_parser("ingest", help="Обновить снимок дисциплин с ВШЭ")
    sub.add_parser("check-api", help="Проверить модель и, если задан ключ, генерацию")
    for command in ("search", "ask"):
        p = sub.add_parser(command, help="Поиск без ключей" if command == "search" else "Ответ через Agent API без Telegram")
        p.add_argument("question")
    args = parser.parse_args()
    args.command = args.command or "run"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(execute(args))
    except (ValueError, AgentAPIError, FileNotFoundError, httpx.HTTPError) as e:
        # HTTPError может содержать URL; здесь только операции без секретов в URL.
        parser.exit(1, f"Ошибка: {e}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
