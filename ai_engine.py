import requests
import time
import os
import logging

logger = logging.getLogger("AI_Engine")

DEEPSEEK_API_URL = "http://localhost:9655/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-any")
DEEPSEEK_MODEL = "deepseek-chat"
REQUEST_TIMEOUT = 1800

MAX_PROMPT_LENGTH = 500000
MAX_RETRIES = 3
RETRY_DELAY = 2


def generate_response(prompt, model=None):
    if model is None:
        model = DEEPSEEK_MODEL

    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.error(f"[AI] Промпт слишком длинный ({len(prompt)} > {MAX_PROMPT_LENGTH})")
        raise ValueError(f"Промпт слишком длинный (максимум {MAX_PROMPT_LENGTH} символов)")

    logger.info(f"[AI] Запрос к DeepSeek: {len(prompt)} символов → model={model}")
    start = time.time()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    text = None
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{DEEPSEEK_API_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                verify=False,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            break
            
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[AI] Попытка {attempt + 1}/{MAX_RETRIES} не удалась: {e}. Повтор через {RETRY_DELAY}с...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"[AI] Все {MAX_RETRIES} попытки подключения к DeepSeek API не удались", exc_info=True)
                raise ConnectionError(f"Ошибка подключения к DeepSeek API после {MAX_RETRIES} попыток") from e

        except (KeyError, IndexError) as e:
            logger.error(f"[AI] Некорректный ответ от DeepSeek API: {e}", exc_info=True)
            raise ValueError("Некорректный ответ от DeepSeek API (см. логи)") from e

    if text is None:
        logger.error("[AI] Не удалось получить ответ от DeepSeek API")
        raise ConnectionError("Не удалось получить ответ от DeepSeek API")
    
    elapsed = time.time() - start
    logger.info(f"[AI] Ответ за {elapsed:.1f} сек: {len(text)} символов")

    text = text.replace("```xml", "").replace("```", "").strip()
    return text


def check_connection():
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        resp = requests.get(f"{DEEPSEEK_API_URL}/models", headers=headers, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_model_info():
    info = {
        "status": "disconnected",
        "server": DEEPSEEK_API_URL,
        "model": DEEPSEEK_MODEL,
    }
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        resp = requests.get(f"{DEEPSEEK_API_URL}/models", headers=headers, timeout=5)
        if resp.status_code == 200:
            info["status"] = "connected"
    except Exception as e:
        logger.warning(f"[AI] Ошибка проверки подключения: {e}")
    return info


def validate_and_fix_xml(generated_xml, example_xml, article_text=None):
    from hypothesis_schema import pipeline_validate_and_fix

    return pipeline_validate_and_fix(
        generated_xml_str=generated_xml,
        example_xml_str=example_xml,
        article_text=article_text,
        llm_generate_fn=generate_response,
    )
