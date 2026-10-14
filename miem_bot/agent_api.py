import asyncio
import json
import logging

import httpx

from .config import Settings


class AgentAPIError(RuntimeError):
    """Содержит только безопасное сообщение, без ключей и тела запроса."""


class AgentAPI:
    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.base_url + "/", timeout=httpx.Timeout(90, connect=15),
            headers={"x-api-key": settings.api_key}, transport=transport,
        )

    async def close(self):
        await self.client.aclose()

    async def complete_json(self, system: str, payload: dict, max_tokens=4096) -> dict:
        # В действующем DTO Agent API эти поля принимаются в camelCase.
        # Общий бюджет включает скрытые reasoning-токены DeepSeek. Даже короткому
        # JSON нужен запас: с лимитом 900 провайдер возвращал content=null.
        body = {"model": self.settings.model, "temperature": 0.1, "maxTokens": max(4096, max_tokens),
                "reasoning": {"effort": "none"},
                "responseFormat": {"type": "json_object"}, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        for attempt in range(3):
            try:
                response = await self.client.post("chat/completions", json=body)
            except httpx.TimeoutException:
                raise AgentAPIError("Agent API не ответил вовремя. Попробуйте ещё раз.") from None
            except httpx.TransportError:
                raise AgentAPIError("Нет соединения с Agent API. Попробуйте позже.") from None
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(1 + attempt)
                continue
            if response.status_code in (401, 403):
                raise AgentAPIError("Agent API отклонил ключ. Проверьте AGENTAPI_KEY в .env.")
            if response.is_error:
                raise AgentAPIError(f"Agent API: ошибка {response.status_code}. Проверьте баланс, модель и параметры в .env.")
            try:
                result = response.json()
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    if attempt < 2 and body["maxTokens"] < 12288:
                        body["maxTokens"] = min(body["maxTokens"] * 2, 12288)
                        logging.getLogger(__name__).warning("Ответ модели оборван по лимиту; повтор с бюджетом %s", body["maxTokens"])
                        continue
                    raise AgentAPIError("Модель не успела завершить ответ. Попробуйте задать более короткий вопрос.")
                content = choice["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("not object")
                return parsed
            except (ValueError, KeyError, TypeError, IndexError, AttributeError):
                if attempt == 0:
                    logging.getLogger(__name__).warning("Модель вернула некорректный JSON; повтор запроса")
                    body["messages"][0]["content"] = system + "\nВерни только один завершённый валидный JSON-объект без Markdown."
                    continue
                raise AgentAPIError("Модель вернула некорректный ответ. Попробуйте переформулировать вопрос.") from None
        raise AgentAPIError("Agent API временно недоступен.")
