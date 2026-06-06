import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI


TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TRITONAI_MODEL = "api-gemma-4-26b"


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent visual navigation system. Your job
is to audit a sub-goal proposed by the Planning Agent before it is forwarded
to the Action Agent for execution in ProcTHOR.

You will receive:
1. Task: the original natural-language goal (e.g., "find a television").
2. Proposed sub-goal: the plan text emitted by the Planning Agent.
3. Map summary: a textual snapshot of the spatial memory built so far
   (nodes visited, objects seen, current location). Per-object distances
   here are the *closest ever observed*, not the current distance.
4. Visual input: a textual description of what the agent currently sees.
5. Action history: the most recent actions taken by the agent.
6. Trajectory (optional): the agent's straight-line distance to the target
   object after each recent action, e.g. "4.98m -> 4.74m -> 4.50m". A
   monotonically decreasing trajectory is good; flat or increasing is bad.
7. Target recency (optional): how many cycles ago the target was last
   detected by the perception model, and the actions executed since that
   sighting. Used by criterion (B) to distinguish "target genuinely out of
   view" from "target temporarily dropped by an unreliable detector while
   still in the forward camera frustum".

Audit the proposal against FOUR criteria, in order. **Reject (approved=false)
if ANY criterion fails.** A reject is not a punishment — it is how you give
the planner a chance to course-correct.

(A) Goal alignment: does the sub-goal make progress toward the task?
(B) Map consistency: is the sub-goal reachable / non-contradictory given the
    map? Specifically: if the sub-goal says "move toward the <target>" but
    the target is NOT in the current visual input AND the agent has no
    inferred direction to it from the map, reject — demand the planner first
    propose a *search* action (e.g., RotateLeft/RotateRight to look around,
    or move to a known landmark) rather than a blind "MoveAhead".
    EXCEPTION (perception-recency carve-out): if the "Target recency" field
    below shows the target was detected within the last 2 perception frames
    AND no rotation actions (RotateLeft/RotateRight) have been executed
    since that last sighting, treat the target as still in the forward
    camera frustum. The perception model is known to drop small/distant
    objects inconsistently between frames; without an intervening rotation,
    a recent sighting remains a valid bearing. In this case, do NOT reject a
    MoveAhead that continues toward the recently-seen target — approve it.
    This carve-out does NOT apply if rotation has happened since the last
    sighting (the prior bearing is stale) or if recency exceeds 2 cycles.
    BLOCKED-NODE RULE: if the Map summary contains a field
    `blocked_actions_at_current_node:` with one or more movement actions
    listed (MoveAhead / MoveBack / MoveLeft / MoveRight) AND the proposed
    sub-goal is movement-shaped (uses verbs like "move", "navigate",
    "approach", "go to", "head toward"), reject. The current node is
    physically obstructed in the direction(s) the agent has tried. In
    `revised_subgoal`, instruct the planner to rotate by 90 degrees or
    more to attempt a fundamentally different heading before any further
    translation.
(C) Non-repetition: does the sub-goal avoid re-doing recent actions or
    re-visiting already-explored locations without new justification?
    NOTE on failure tags: entries in Action history that end with
    "[FAILED: <reason>]" mean the simulator rejected the action (typically
    a collision with furniture) and the agent did NOT move. Two or more
    consecutive [FAILED] entries (regardless of which specific action
    was attempted) are a strong signal that the agent is wedged at the
    current position. Reject any movement-shaped sub-goal in this state.
    In `revised_subgoal`, instruct the planner to rotate by 90 degrees
    or more to face a different direction, then re-plan from the new
    view.
(D) Progress trend (HARD RULE): if the Trajectory is provided and the
    agent has successfully executed at least 3 *movement* actions
    (MoveAhead, Teleport) in the recent history, AND the distance to
    target has NOT strictly decreased over the last 3 such movements
    (i.e., the most recent distance is >= the distance from 3 movements
    ago), reject. The agent is drifting. In `revised_subgoal`, instruct
    the planner to STOP moving forward, ROTATE to re-acquire the target
    visually, and re-plan from what it then sees.
    Note: action history entries marked "[FAILED: ...]" did NOT execute
    (the simulator rejected them) and do NOT count toward the 3-movement
    threshold here. Use (C) to handle repeated [FAILED] entries.

Respond with a single JSON object and nothing else:
{
  "approved": <true|false>,
  "reason": "<one-sentence explanation; cite the criterion letter (A/B/C/D)>",
  "revised_subgoal": "<if rejected, a corrected sub-goal; otherwise null>"
}
""".strip()


class CriticAgent:
    def __init__(self, model, client):
        self.model = model
        self.client = client

    @classmethod
    def from_openai(cls, model=DEFAULT_OPENAI_MODEL, api_key=None):
        if api_key is None:
            with open("key.txt", "r") as f:
                api_key = f.read().strip()
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        return cls(model=model, client=client)

    @classmethod
    def from_tritonai(cls, model=DEFAULT_TRITONAI_MODEL, api_key=None):
        load_dotenv(cls._repo_env_path())
        key = api_key or os.environ.get("TRITONAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "TritonAI API key not found. Set TRITONAI_API_KEY (or OPENAI_API_KEY), "
                "e.g. `export TRITONAI_API_KEY=$(cat api-key.txt)`."
            )
        client = OpenAI(base_url=TRITONAI_BASE_URL, api_key=key)
        return cls(model=model, client=client)

    def review(
        self,
        task,
        proposed_subgoal,
        map_summary="",
        action_history=None,
        perception_description="",
        distance_history=None,
        target_type=None,
        cycles_since_target_seen=None,
        actions_since_target_seen=None,
    ):
        prompt = self._build_prompt(
            task,
            proposed_subgoal,
            map_summary,
            action_history or [],
            perception_description,
            distance_history or [],
            target_type,
            cycles_since_target_seen,
            actions_since_target_seen,
        )
        raw = self._create_text_response(prompt)
        return self._parse_verdict(raw)

    def _create_text_response(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _build_prompt(
        self,
        task,
        proposed_subgoal,
        map_summary,
        action_history,
        perception_description,
        distance_history,
        target_type,
        cycles_since_target_seen=None,
        actions_since_target_seen=None,
    ):
        # Cap to last 8 entries to keep the prompt focused.
        recent_actions = action_history[-8:] if action_history else []
        history_text = "\n".join(f"- {a}" for a in recent_actions) if recent_actions else "(none)"

        if distance_history:
            recent_dists = distance_history[-9:]  # one more than actions: pre-action distances + final
            dist_strs = [f"{d:.2f}m" if d is not None else "n/a" for d in recent_dists]
            trajectory_text = " -> ".join(dist_strs)
            if target_type:
                trajectory_text = f"(distance to '{target_type}') {trajectory_text}"
        else:
            trajectory_text = "(not provided)"

        recency_text = self._format_recency(
            cycles_since_target_seen, actions_since_target_seen, target_type
        )

        return (
            f"{CRITIC_SYSTEM_PROMPT}\n\n"
            f"Task:\n{task}\n\n"
            f"Proposed sub-goal:\n{proposed_subgoal}\n\n"
            f"Map summary:\n{map_summary or '(empty)'}\n\n"
            f"Visual input:\n{perception_description or '(not provided)'}\n\n"
            f"Action history (most recent {len(recent_actions)}):\n{history_text}\n\n"
            f"Trajectory:\n{trajectory_text}\n\n"
            f"Target recency:\n{recency_text}\n"
        )

    @staticmethod
    def _format_recency(cycles_since, actions_since, target_type):
        if cycles_since is None:
            return "(target has not yet been detected by perception)"

        actions_since = actions_since or []
        has_rotation = any(
            ("RotateLeft" in a) or ("RotateRight" in a) for a in actions_since
        )
        target_label = f"'{target_type}'" if target_type else "target"

        if cycles_since == 0:
            return f"{target_label} detected in the CURRENT perception frame."

        rotation_note = (
            "rotation HAS occurred since last sighting (prior bearing is stale)"
            if has_rotation
            else "NO rotation since last sighting (prior bearing still valid)"
        )
        actions_text = ", ".join(actions_since) if actions_since else "(none)"
        return (
            f"{target_label} last detected {cycles_since} cycle(s) ago. "
            f"Actions executed since last sighting: {actions_text}. "
            f"{rotation_note}."
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

    @staticmethod
    def _repo_env_path():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))


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
