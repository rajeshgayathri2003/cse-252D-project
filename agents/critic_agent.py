import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI


TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TRITONAI_MODEL = "api-mistral-small-3.2-2506"


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent visual navigation system. Your job
is to audit a sub-goal proposed by the Planning Agent before it is forwarded
to the Action Agent for execution in ProcTHOR.

You will receive:
1. Task: the original natural-language goal (e.g., "find a television").
2. Proposed sub-goal: the plan text emitted by the Planning Agent.
3. Map summary: a textual snapshot of the spatial memory built so far
   (nodes visited, objects seen, current location).
4. Action history: the most recent actions taken by the agent.

Audit the proposal against three criteria:
- Goal alignment: does the sub-goal make progress toward the task?
- Map consistency: is the sub-goal reachable / non-contradictory given the map?
- Non-repetition: does the sub-goal avoid re-doing recent actions or
  re-visiting already-explored locations without new justification?

Respond with a single JSON object and nothing else:
{
  "approved": <true|false>,
  "reason": "<one-sentence explanation>",
  "revised_subgoal": "<if rejected, a corrected sub-goal; otherwise null>"
}
""".strip()


class CriticAgent:
    def __init__(self, model, client):
        self.model = model
        self.client = client

    @classmethod
    def from_openai(cls, model=DEFAULT_OPENAI_MODEL, api_key=None):
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        return cls(model=model, client=client)

    @classmethod
    def from_tritonai(cls, model=DEFAULT_TRITONAI_MODEL, api_key=None):
        load_dotenv()  # pulls TRITONAI_API_KEY from .env at repo root if present
        key = api_key or os.environ.get("TRITONAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "TritonAI API key not found. Set TRITONAI_API_KEY (or OPENAI_API_KEY), "
                "e.g. `export TRITONAI_API_KEY=$(cat api-key.txt)`."
            )
        client = OpenAI(base_url=TRITONAI_BASE_URL, api_key=key)
        return cls(model=model, client=client)

    def review(self, task, proposed_subgoal, map_summary="", action_history=None):
        prompt = self._build_prompt(task, proposed_subgoal, map_summary, action_history or [])
        raw = self.client.responses.create(model=self.model, input=prompt).output_text
        return self._parse_verdict(raw)

    def _build_prompt(self, task, proposed_subgoal, map_summary, action_history):
        history_text = "\n".join(f"- {a}" for a in action_history) if action_history else "(none)"
        return (
            f"{CRITIC_SYSTEM_PROMPT}\n\n"
            f"Task:\n{task}\n\n"
            f"Proposed sub-goal:\n{proposed_subgoal}\n\n"
            f"Map summary:\n{map_summary or '(empty)'}\n\n"
            f"Action history:\n{history_text}\n"
        )

    def _parse_verdict(self, raw):
        # On malformed output, default to rejected so the planner re-plans
        # rather than the loop silently approving a bad sub-goal.
        verdict = self._extract_json(raw)
        if verdict is None:
            return {
                "approved": False,
                "reason": f"Critic response was not valid JSON: {raw!r}",
                "revised_subgoal": None,
            }

        return {
            "approved": bool(verdict.get("approved", False)),
            "reason": str(verdict.get("reason", "")),
            "revised_subgoal": verdict.get("revised_subgoal"),
        }

    @staticmethod
    def _extract_json(raw):
        if not isinstance(raw, str):
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Strip ```json ... ``` or ``` ... ``` code fences that small LLMs love.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    # Smoke test against TritonAI — student-accessible, free for UCSD.
    # For OpenAI instead: critic = CriticAgent.from_openai()
    critic = CriticAgent.from_tritonai()
    verdict = critic.review(
        task="find a television",
        proposed_subgoal="MoveAhead toward the doorway leading into the living room",
        map_summary="Spatial map summary:\n- nodes: 4\n- edges: 6\n- known_objects:\n  - Sofa: (1.0, 2.0, 0)",
        action_history=["MoveAhead", "RotateRight", "MoveAhead"],
    )
    print(verdict)
