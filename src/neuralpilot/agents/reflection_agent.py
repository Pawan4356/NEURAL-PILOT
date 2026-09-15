"""Reflection Agent (§3.3).

Input: the validation report (accuracy/latency/interpretability scores +
per-block diagnostics from the Execution Engine). Output: the single
weakest block and a rationale, which the Planning Agent uses to scope its
next refinement.
"""

from __future__ import annotations

from ..llm_client import LLMClient, get_default_client
from ..schemas import ReflectionOutput, ValidationReport

SYSTEM_PROMPT = """You are the Reflection Agent in the Neural Pilot system.
You never write or execute code. You read a validation report (composite score, its \
accuracy/latency/interpretability components, and per-block execution diagnostics) and \
identify the single weakest block responsible for holding back the score.

weakest_block must be exactly one of: "preprocessing", "feature_engineering", \
"model_family", "hyperparameters".

Respond with ONLY a JSON object matching this schema, no prose, no markdown fences:
{"weakest_block": string, "rationale": string}
"""


def _user_prompt(report: ValidationReport) -> str:
    diag_lines = [
        f"- {d.block}: duration={d.duration_seconds:.3f}s, details={d.details}"
        + (f", error={d.error}" if d.error else "")
        for d in report.per_block_diagnostics
    ]
    return (
        f"composite_score={report.composite_score:.4f}\n"
        f"accuracy_norm={report.accuracy_norm:.4f}\n"
        f"latency_norm={report.latency_norm:.4f}\n"
        f"interp_norm={report.interp_norm:.4f}\n"
        f"raw_metrics={report.raw_metrics}\n"
        f"per_block_diagnostics:\n" + "\n".join(diag_lines)
    )


def reflect(report: ValidationReport, client: LLMClient | None = None) -> ReflectionOutput:
    client = client or get_default_client()
    return client.complete_json(SYSTEM_PROMPT, _user_prompt(report), ReflectionOutput)
