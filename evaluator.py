import json
from typing import List, Dict, Any

class LLMEvaluator:
    """
    A utility class to evaluate LLM responses based on predefined rubrics.
    Useful for RLHF pipelines and automated model grading.
    """
    
    def __init__(self, criteria_weights: Dict[str, float] = None):
        self.weights = criteria_weights or {
            "helpfulness": 0.4,
            "relevance": 0.3,
            "clarity": 0.3
        }

    def score_response(self, response: str, prompt: str, scores: Dict[str, int]) -> float:
        try:
            final_score = sum(
                scores.get(criterion, 0) * weight 
                for criterion, weight in self.weights.items()
            )
            return round(final_score, 2)
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0

    def batch_evaluate(self, dataset: List[Dict[str, Any]]) -> str:
        results = []
        for item in dataset:
            score = self.score_response(
                item.get("response", ""),
                item.get("prompt", ""),
                item.get("scores", {})
            )
            results.append({"id": item.get("id"), "final_score": score})
        return json.dumps({"batch_results": results}, indent=4)
