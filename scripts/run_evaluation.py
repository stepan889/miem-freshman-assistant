"""Реальные 30 вопросов к тому же Assistant, который обслуживает Telegram.

Запуск: .venv/bin/python scripts/run_evaluation.py
Один ответ на сценарий, без перезапуска неудачных сценариев ради хорошего отчёта.
"""
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miem_bot.agent_api import AgentAPI
from miem_bot.assistant import Assistant
from miem_bot.config import Settings, DATA_FILE
from miem_bot.retrieval import KnowledgeBase


async def main():
    settings = Settings.load()
    settings.require_keys()
    kb = KnowledgeBase()
    api = AgentAPI(settings)
    assistant = Assistant(kb, api)
    questions = json.loads((ROOT / "evaluation/questions.json").read_text())
    path = ROOT / "deliverables/testing_raw.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise SystemExit("testing_raw.json уже существует. Сохраните предыдущий прогон перед новым.")
    results = []
    async def request(case):
        history = []
        if case.get("history_from"):
            previous = next(r for r in results if r["id"] == case["history_from"])
            history = [{"role": "user", "content": previous["question"]},
                       {"role": "assistant", "content": previous["answer"][:2000]}]
        selected = next((c["id"] for c in kb.courses if c["title"] == case.get("selected_course_title")), None)
        started = time.monotonic()
        result = dict(case)
        try:
            answer = await assistant.ask(case["question"], history=history, selected_course=selected)
            result.update({"answer": answer.text, "sources": [
                {"number": i, "course": c.course, "section": c.section, "url": c.url}
                for i, c in enumerate(answer.sources, 1)], "error": None})
        except Exception as e:
            message = str(e)
            for secret in (settings.api_key, settings.telegram_token):
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            result.update({"answer": message, "sources": [], "error": type(e).__name__})
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        result["tested_at"] = datetime.now(ZoneInfo("Europe/Moscow")).isoformat()
        return result

    def save():
        output = {"model": settings.model, "base_url": settings.base_url,
                  "method": "Прямой вызов Assistant.ask, используемого Telegram-ботом; реальные запросы Agent API, без имитации ответов.",
                  "snapshot_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
                  "academic_year": "2026/2027", "study_year": 1,
                  "results": sorted(results, key=lambda r: r["id"])}
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    try:
        # Два запроса одновременно оставляют запас для запущенного Telegram-бота.
        for start in range(0, len(questions), 2):
            batch = await asyncio.gather(*(request(c) for c in questions[start:start+2]))
            results.extend(batch)
            save()
            for r in batch:
                print(f"{r['id']:02} {'ERROR' if r['error'] else 'OK'} {r['elapsed_seconds']} сек. {len(r['sources'])} источников", flush=True)
    finally:
        await api.close()
    print(f"Сохранено {len(results)} ответов в {path}")


if __name__ == "__main__":
    asyncio.run(main())
