import time
from typing import Dict, Any, List
from src.telemetry.logger import logger

class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs.
    """
    def __init__(self):
        self.session_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_estimate": self._calculate_cost(model, usage) # Mock cost calculation
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculates real pricing logic based on model names.
        Prices represent standard rates per 1,000,000 tokens.
        """
        model_lower = model.lower()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Local model runs free of charge on CPU/GPU
        if "phi-3" in model_lower or "local" in model_lower:
            return 0.0
            
        # GPT-4o: $5.00 per 1M prompt tokens, $15.00 per 1M completion tokens
        elif "gpt-4o" in model_lower:
            cost = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)
            return round(cost, 6)
            
        # Gemini 1.5 Flash: $0.075 per 1M prompt tokens, $0.30 per 1M completion tokens (approximate pay-as-you-go)
        elif "gemini" in model_lower:
            cost = (prompt_tokens * 0.000000075) + (completion_tokens * 0.0000003)
            return round(cost, 8)
            
        # Default pricing fallback
        else:
            return round((usage.get("total_tokens", 0) / 1000) * 0.002, 6)

# Global tracker instance
tracker = PerformanceTracker()

