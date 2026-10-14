import json

import httpx
import pytest

from miem_bot.agent_api import AgentAPI, AgentAPIError
from miem_bot.assistant import Assistant
from miem_bot.config import Settings
from miem_bot.retrieval import KnowledgeBase


@pytest.fixture
def kb():
    return KnowledgeBase()


def course_id(kb, title):
    return next(c["id"] for c in kb.courses if c["title"] == title)


def test_literature_and_grading_stay_in_selected_variant(kb):
    cid = course_id(kb, "Алгебра и геометрия")
    chunks = kb.context_for_plan("оценка", [cid], ["Промежуточная аттестация", "Элементы контроля"])
    assert all(c.course_id == cid for c in chunks)
    formulas = " ".join(c.text for c in chunks if c.section == "Промежуточная аттестация")
    assert "0.275" in formulas and "2026/2027 4th module" in formulas
    assert "0.2 * 2026/2027 2nd module" in formulas
    literature = kb.context_for_plan("книги", [cid], ["Рекомендуемая основная литература", "Рекомендуемая дополнительная литература"])
    assert any("Беклемишев" in c.text for c in literature)
    assert any("Попов" in c.text for c in literature)


def test_comparison_has_both_courses(kb):
    ids = [course_id(kb, t) for t in ("Алгебра и геометрия", "Дискретная математика")]
    chunks = kb.context_for_plan("связь", ids, ["Аннотация", "Содержание учебной дисциплины"])
    assert {c.course_id for c in chunks} == set(ids)
    assert all(any(c.course_id == i and c.section == "Аннотация" for c in chunks) for i in ids)


def test_topic_search(kb):
    chunks = kb.search("графы деревья связность", limit=8)
    assert any("Дискретная математика" in c.course for c in chunks)


def test_citations_are_checked_and_deduplicated(kb):
    c = next(c for c in kb.chunks if c.section == "Промежуточная аттестация")
    answer = Assistant.validate_answer({"answer": "Формула [S2]. Модуль [S1].", "source_ids": ["S2", "S1"]}, {"S1": c, "S2": c})
    assert answer.text == "Формула [1]. Модуль [1]."
    assert len(answer.sources) == 1
    assert "hse.ru" in answer.plain()


@pytest.mark.parametrize("reply", [
    {"answer": "Факт [S99]", "source_ids": ["S99"]},
    {"answer": "Факт [S1]", "source_ids": []},
    {"answer": "Факт", "source_ids": ["S1"]},
    {"answer": "<b>Факт</b> [S1]", "source_ids": ["S1"]},
    {"answer": "https://invented.example [S1]", "source_ids": ["S1"]},
])
def test_invalid_citations_rejected(kb, reply):
    with pytest.raises(AgentAPIError):
        Assistant.validate_answer(reply, {"S1": kb.chunks[0]})


def test_no_uncited_factual_answer(kb):
    answer = Assistant.validate_answer({"answer": "Придуманная формула 0.7*экзамен", "source_ids": []}, {})
    assert "0.7" not in answer.text
    assert not answer.sources


async def test_real_client_request_contract_and_full_rag_flow(kb):
    calls = []
    cid = course_id(kb, "Алгебра и геометрия")

    def handle(request):
        assert str(request.url) == "https://api.agentapi.ru/v1/ai/chat/completions"
        assert request.headers["x-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["model"] == "deepseek/deepseek-v4-flash"
        assert body["responseFormat"] == {"type": "json_object"}
        assert body["reasoning"] == {"effort": "none"}
        assert body["maxTokens"] >= 4096
        assert "response_format" not in body and "max_tokens" not in body
        payload = json.loads(body["messages"][1]["content"])
        calls.append(payload)
        if "catalog" in payload:
            reply = {"mode": "answer", "course_ids": [int(cid)], "sections": ["Промежуточная аттестация"], "query": "формула оценки"}
        else:
            evidence = next(c for c in payload["evidence"] if c["section"] == "Промежуточная аттестация")
            assert "0.275" in evidence["text"]
            reply = {"answer": f"Во втором модуле контрольная имеет вес 0.275 [{evidence['id']}].", "source_ids": [evidence["id"]]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply, ensure_ascii=False)}}]})

    api = AgentAPI(Settings(api_key="test-key"), transport=httpx.MockTransport(handle))
    try:
        answer = await Assistant(kb, api).ask("Как считается оценка по обычной алгебре?", selected_course=cid)
        assert len(calls) == 2
        assert answer.sources[0].url.endswith("#intermediate_certification")
        assert "0.275 [1]" in answer.text
    finally:
        await api.close()


class FakeAPI:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def complete_json(self, system, payload, **kwargs):
        self.calls.append(payload)
        return self.response


async def test_clarification_uses_history_without_generation(kb):
    api = FakeAPI({"mode": "clarify", "clarification": "Какую дисциплину вы имеете в виду?"})
    history = [{"role": "user", "content": "Алгебра и геометрия"}]
    answer = await Assistant(kb, api).ask("А литература?", history)
    assert len(api.calls) == 1
    assert api.calls[0]["history"] == history
    assert not answer.sources


async def test_out_of_scope_never_generates_facts(kb):
    api = FakeAPI({"mode": "out_of_scope"})
    answer = await Assistant(kb, api).ask("Какая завтра погода?")
    assert "1 курса" in answer.text
    assert len(api.calls) == 1


async def test_hallucinated_course_id_is_rejected(kb):
    api = FakeAPI({"mode": "answer", "course_ids": ["invented"], "sections": []})
    with pytest.raises(AgentAPIError):
        await Assistant(kb, api).ask("Предмет")


async def test_api_key_error_does_not_leak_secrets():
    def handle(request):
        return httpx.Response(401, json={"message": "agentapi-secret-do-not-log"})
    api = AgentAPI(Settings(api_key="agentapi-secret-do-not-log"), transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(AgentAPIError) as e:
            await api.complete_json("system", {})
        assert "secret" not in str(e.value)
    finally:
        await api.close()


async def test_reasoning_consumes_budget_then_retries_complete_json():
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(201, json={"choices": [{"finish_reason": "length", "message": {"content": None}}],
                                           "usage": {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}})
        return httpx.Response(201, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})

    api = AgentAPI(Settings(api_key="test-key"), transport=httpx.MockTransport(handle))
    try:
        assert await api.complete_json("system", {}, max_tokens=900) == {"ok": True}
        assert [body["maxTokens"] for body in requests] == [4096, 8192]
        assert all(body["reasoning"] == {"effort": "none"} for body in requests)
    finally:
        await api.close()


async def test_truncation_retries_are_bounded():
    budgets = []

    def handle(request):
        budgets.append(json.loads(request.content)["maxTokens"])
        return httpx.Response(201, json={"choices": [{"finish_reason": "length", "message": {"content": None}}]})

    api = AgentAPI(Settings(api_key="test-key"), transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(AgentAPIError, match="не успела завершить"):
            await api.complete_json("system", {})
        assert budgets == [4096, 8192, 12288]
    finally:
        await api.close()


async def test_one_malformed_json_retries_without_accepting_bad_result():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        content = '{"ok":' if len(requests) == 1 else '\n```json\n{"ok":true}\n```\n'
        return httpx.Response(201, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    api = AgentAPI(Settings(api_key="test-key"), transport=httpx.MockTransport(handle))
    try:
        assert await api.complete_json("system", {}) == {"ok": True}
        assert len(requests) == 2
        assert "валидный JSON" in requests[1]["messages"][0]["content"]
    finally:
        await api.close()
