import os
import time
import google.generativeai as genai
from typing import Dict, Any, Optional, Generator
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, stop: Optional[list] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # In Gemini, system instruction is passed during model initialization or as a prefix
        # For simplicity in this lab, we'll prepend it if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        import google.api_core.exceptions
        max_retries = 5
        retry_delay = 12.0
        response = None
        
        # Config stop sequences if provided
        gen_config = {"stop_sequences": stop} if stop else None
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(full_prompt, generation_config=gen_config)
                break
            except google.api_core.exceptions.ResourceExhausted as e:
                if attempt == max_retries - 1:
                    logger.error(f"Gemini API rate limit exceeded after {max_retries} attempts.")
                    raise e
                logger.info(f"Gemini API rate limit hit. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay *= 1.5
            except Exception as e:
                # Catch other transient network errors that might look like 429/ResourceExhausted
                if "quota" in str(e).lower() or "limit" in str(e).lower() or "429" in str(e).lower():
                    if attempt == max_retries - 1:
                        raise e
                    logger.info(f"Transient error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise e

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Gemini usage data is in response.usage_metadata
        content = response.text
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "google"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        response = self.model.generate_content(full_prompt, stream=True)
        for chunk in response:
            yield chunk.text
