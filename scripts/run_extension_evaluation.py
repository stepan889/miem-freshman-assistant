"""Дополнительные рабочие сценарии для итогового демонстрационного набора."""
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
    api = AgentAPI(Settings.load())
    kb = KnowledgeBase()
    assistant = Assistant(kb, api)
    cases = [
        {"id":33,"group":"Возможности","question":"В каких модулях читается «История России» и сколько кредитов у этой дисциплины?",
         "criterion":"Назвать 1–4 модули и 4 кредита из карточки «История России»."},
        {"id":34,"group":"Возможности","question":"Какие основные разделы перечислены в содержании дисциплины «Основы российской государственности»?",
         "criterion":"Перечислить разделы содержания с источником."},
        {"id":35,"group":"Возможности","question":"В каком модуле проводится независимый экзамен по ИИ-грамотности и сколько у него кредитов?",
         "criterion":"Назвать 3-й модуль и отметить, что количество кредитов в карточке не указано."},
        {"id":36,"group":"Возможности","question":"Какая основная литература рекомендована по алгебре и геометрии?",
         "selected_course_title":"Алгебра и геометрия","criterion":"В режиме выбранной обычной алгебры назвать основную литературу этой программы."},
        {"id":37,"group":"Возможности","question":"А какая дополнительная литература?",
         "history_from":36,"selected_course_title":"Алгебра и геометрия","criterion":"Продолжить диалог в выбранной дисциплине и назвать дополнительную литературу обычной алгебры."},
        {"id":38,"group":"Возможности","question":"Я выбрал обычную информатику в меню. Какие формулы оценки действуют в первом и втором модулях?",
         "selected_course_title":"Информатика","criterion":"Вернуть обе формулы обычной информатики с коэффициентом экзамена 0.35."},
    ]
    if len(sys.argv)>1:
        ids={int(x) for x in sys.argv[1:]}
        cases=[c for c in cases if c['id'] in ids]
    results = []
    try:
        for c in cases:
            selected = next((x['id'] for x in kb.courses if x['title']==c.get('selected_course_title')),None)
            history = []
            if c.get('history_from'):
                previous = next(x for x in results if x['id']==c['history_from'])
                history=[{'role':'user','content':previous['question']},{'role':'assistant','content':previous['answer'][:2000]}]
            started=time.monotonic()
            try:
                answer=await assistant.ask(c['question'],history,selected)
                result={**c,'answer':answer.text,'sources':[{'number':i,'course':x.course,'section':x.section,'url':x.url} for i,x in enumerate(answer.sources,1)],'error':None}
            except Exception as e:
                result={**c,'answer':str(e),'sources':[],'error':type(e).__name__}
            result.update(elapsed_seconds=round(time.monotonic()-started,2),tested_at=datetime.now(ZoneInfo('Europe/Moscow')).isoformat())
            results.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        await api.close()
    target=ROOT/'deliverables/testing_extensions_raw.json'
    old=target if target.exists() else ROOT/'evaluation/runs/testing_extensions_raw.json'
    merged={r['id']:r for r in json.loads(old.read_text())['results']} if old.exists() else {}
    merged.update({r['id']:r for r in results})
    target.write_text(json.dumps({'results':list(merged.values())},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':asyncio.run(main())
