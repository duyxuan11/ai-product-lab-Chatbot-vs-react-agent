import os
import sys
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

class BaselineChatbot:
    """
    A baseline chatbot that sends queries directly to the LLM.
    Used for comparing accuracy/latency against the ReAct agent.
    """
    def __init__(self):
        provider_name = os.getenv("DEFAULT_PROVIDER", "openai").lower()
        model_name = os.getenv("DEFAULT_MODEL", "gpt-4o")
        
        logger.info(f"Initializing Baseline Chatbot with provider={provider_name}, model={model_name}")
        
        if provider_name == "openai":
            self.llm = OpenAIProvider(model_name=model_name, api_key=os.getenv("OPENAI_API_KEY"))
        elif provider_name in ["gemini", "google"]:
            self.llm = GeminiProvider(model_name=model_name, api_key=os.getenv("GEMINI_API_KEY"))
        elif provider_name == "local":
            model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
            self.llm = LocalProvider(model_path=model_path)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def run(self, user_input: str) -> str:
        logger.log_event("CHATBOT_START", {"input": user_input, "model": self.llm.model_name})
        
        system_prompt = "You are a helpful nutrition assistant. Answer the user's questions to the best of your ability."
        
        response_dict = self.llm.generate(user_input, system_prompt=system_prompt)
        content = response_dict.get("content", "").strip()
        
        # Track metric
        tracker.track_request(
            provider=response_dict.get("provider", "openai"),
            model=self.llm.model_name,
            usage=response_dict.get("usage", {}),
            latency_ms=response_dict.get("latency_ms", 0)
        )
        
        logger.log_event("CHATBOT_END", {"status": "SUCCESS"})
        return content

if __name__ == "__main__":
    chatbot = BaselineChatbot()
    print("Baseline Chatbot initialized. Type 'exit' to quit.\n")
    while True:
        try:
            user_in = input("User: ")
            if user_in.strip().lower() in ["exit", "quit"]:
                break
            response = chatbot.run(user_in)
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            break
