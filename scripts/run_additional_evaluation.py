"""Два дополнительных сценария интерфейса: выбор дисциплины и история."""
import asyncio
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
from miem_bot.config import Settings
from miem_bot.retrieval import KnowledgeBase


async def main():
    settings = Settings.load()
    api = AgentAPI(settings)
    kb = KnowledgeBase()
    assistant = Assistant(kb, api)
    baseline = json.loads((ROOT / "deliverables/testing_raw.json").read_text())
    previous = next(r for r in baseline["results"] if r["id"] == 3)
    cases = [
        {"id":31,"group":"Возможности","question":"Какая формула оценивания у обычного курса информатики?",
         "selected_course_title":"Информатика","criterion":"После выбора «Информатика» в меню вернуть её формулу без лишнего уточнения."},
        {"id":32,"group":"Возможности","question":"А сколько у этой дисциплины кредитов и кто её преподаёт?",
         "history_from":3,"criterion":"По истории вопроса о математическом анализе определить предмет, найти кредиты и преподавателей."},
    ]
    results = []
    try:
        for c in cases:
            started = time.monotonic()
            selected = next((x["id"] for x in kb.courses if x["title"] == c.get("selected_course_title")), None)
            history = ([{"role":"user","content":previous["question"]},
                        {"role":"assistant","content":previous["answer"][:2000]}] if c.get("history_from") else [])
            try:
                answer = await assistant.ask(c["question"], history, selected)
                result = {**c,"answer":answer.text,"sources":[{"number":i,"course":s.course,"section":s.section,"url":s.url}
                           for i,s in enumerate(answer.sources,1)],"error":None}
            except Exception as e:
                result = {**c,"answer":str(e),"sources":[],"error":type(e).__name__}
            result.update(elapsed_seconds=round(time.monotonic()-started,2), tested_at=datetime.now(ZoneInfo("Europe/Moscow")).isoformat())
            results.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        await api.close()
    (ROOT / "deliverables/testing_additional_raw.json").write_text(json.dumps({"results":results},ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    asyncio.run(main())
