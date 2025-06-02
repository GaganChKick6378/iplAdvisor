from langsmith import Client
import config
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

class LangSmithMonitor:
    def __init__(self):
        self.client = Client(
            api_key=config.LANGCHAIN_API_KEY,
            api_url=config.LANGCHAIN_ENDPOINT
        )
        self.project_name = config.LANGCHAIN_PROJECT

    def log_run(self,
                run_type: str,
                inputs: Dict[str, Any],
                outputs: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None,
                metrics: Optional[Dict[str, float]] = None,
                parent_run_id: Optional[str] = None
               ) -> str:
        """
        Log a run to LangSmith, optionally nesting under a parent_run_id.
        """
        try:
            run_id = str(uuid.uuid4())
            start_time = datetime.now()

            run_data = {
                "id": run_id,
                "name": f"{run_type}_{int(time.time())}",
                "run_type": "chain",
                "inputs": inputs,
                "project_name": self.project_name,
                "start_time": start_time
            }

            if parent_run_id:
                run_data["parent_run_id"] = parent_run_id

            if error:
                run_data.update({
                    "outputs": {"error": str(error)},
                    "error": str(error),
                    "end_time": datetime.now(),
                })
            else:
                run_data.update({
                    "outputs": outputs or {},
                    "end_time": datetime.now(),
                })

            self.client.create_run(**run_data)

            if metrics:
                for metric_name, metric_value in metrics.items():
                    try:
                        self.client.create_feedback(
                            run_id=run_id,
                            key=f"metric_{metric_name}",
                            score=float(metric_value),
                            comment=f"Automated metric: {metric_name}"
                        )
                    except Exception as feedback_error:
                        print(f"Warning: Could not log metric {metric_name}: {feedback_error}")

            return run_id

        except Exception as logging_error:
            print(f"Warning: LangSmith logging failed: {logging_error}")
            return str(uuid.uuid4())

    def log_simple_event(self, event_name: str, data: Dict[str, Any]):
        """
        Simple event logging without run tracking.
        """
        try:
            event_data = {
                "event_name": event_name,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "project_name": self.project_name
            }
            run_id = str(uuid.uuid4())
            self.client.create_run(
                id=run_id,
                name=f"event_{event_name}_{int(time.time())}",
                run_type="chain",
                inputs=event_data,
                outputs={"status": "logged"},
                project_name=self.project_name,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
        except Exception as e:
            print(f"Warning: Event logging failed: {e}")

    def calculate_confidence(self,
                            recommendation: str,
                            data_quality: float,
                            data_recency: float,
                            data_relevance: float
                           ) -> float:
        """
        Calculate confidence score for a recommendation.
        """
        uncertainty_phrases = [
            "uncertain", "unclear", "might", "may", "could be",
            "possibly", "perhaps", "not enough data", "limited information"
        ]

        certainty_factor = 1.0
        for phrase in uncertainty_phrases:
            if phrase in recommendation.lower():
                certainty_factor *= 0.9

        confidence = (
            0.4 * data_quality +
            0.3 * data_recency +
            0.3 * data_relevance
        ) * certainty_factor

        return min(max(confidence, 0.0), 1.0)

    def get_confidence_label(self, confidence: float) -> str:
        """
        Convert numerical confidence to a human‐readable label.
        """
        if confidence >= config.HIGH_CONFIDENCE:
            return "High Confidence"
        elif confidence >= config.MEDIUM_CONFIDENCE:
            return "Medium Confidence"
        elif confidence >= config.LOW_CONFIDENCE:
            return "Low Confidence"
        else:
            return "Very Low Confidence"

class SimpleLangSmithMonitor(LangSmithMonitor):
    """
    A simplified subclass that falls back to printing if the client fails.
    Inherits log_run with parent_run_id support.
    """
    def __init__(self):
        try:
            super().__init__()
            self.enabled = True
        except Exception as e:
            print(f"Warning: LangSmith initialization failed: {e}")
            self.enabled = False

    def log_run(self,
                run_type: str,
                inputs: Dict[str, Any],
                outputs: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None,
                metrics: Optional[Dict[str, float]] = None,
                parent_run_id: Optional[str] = None
               ) -> str:
        """
        Override: if disabled, just print. Otherwise behave exactly like parent.
        """
        if not self.enabled:
            print(f"[MONITOR] {run_type}: inputs={inputs}, parent_run_id={parent_run_id}")
            if outputs:
                print(f"[MONITOR OUTPUTS] {outputs}")
            if error:
                print(f"[MONITOR ERROR] {error}")
            if metrics:
                print(f"[MONITOR METRICS] {metrics}")
            return str(uuid.uuid4())

        return super().log_run(
            run_type=run_type,
            inputs=inputs,
            outputs=outputs,
            error=error,
            metrics=metrics,
            parent_run_id=parent_run_id
        )
