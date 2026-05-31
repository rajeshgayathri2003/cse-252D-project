# Planning Agent (`plan/base.py`)

This module implements an early prototype of the **Planning Agent**, one of the
five cooperating agents described in our system architecture (see
`report/week5.tex`, §System Architecture). In the full framework, the planner
sits at the center of a closed simulator/perception–reasoning–action loop: it
fuses a local semantic snapshot from the Perception Agent with a long-horizon
topological path from the Spatial Memory Agent, then proposes the next sub-goal
for the Critic Agent to audit before the Action Agent executes it in ProcTHOR.

The shared agent scaffolding (`BaseAgentCore`, `AgentBase`) lives in
`agents/base.py`, and the Spatial Memory Agent it composes with lives in
`agents/mapping_agent.py`.

## What `plan/base.py` currently provides

The file defines a single class, `PlanningAgent`, that bundles together:

1. **LLM interface.** An OpenAI client is wrapped in `__call__`, which forwards
   a prompt string to the configured chat model and returns the response text.
   The `PROMPT` constant defines the system role: a planning agent that decides
   the next action from a task description, visual input, and a graph/top-down
   view of the environment.

2. **Mapping Agent composition.** The constructor now takes an optional
   `mapping_agent: MappingAgent` (defaulting to a fresh instance). The planner
   no longer owns the spatial graph itself. Following the architecture diagram,
   the Mapping Agent is pose-first: AI2-THOR/ProcTHOR provides `(x, y, z)` and
   yaw directly to `MappingAgent.update(...)` / `update_pose(...)`, while
   perception remains a separate RGB branch. The planner queries the mapping
   agent for a textual summary via `mapping_agent.get_context_string()` whenever
   it builds a prompt.

3. **Environment introspection via AI2-THOR.** The agent holds a
   `ai2thor.controller.Controller` and exposes two utilities for inspecting the
   scene:
   - `get_reachable_positions()` — queries the simulator for all grid cells the
     agent can stand on.
   - `get_top_down_frame()` — adds a third-party camera high above the scene
     and returns a PIL image of the resulting top-down view, intended as the
     `top_down_view` input to `generate_plan`.

4. **Legacy spatial graph construction.** `create_graph(reachable_positions)`
   builds a `networkx.Graph` of reachable `(x, z)` cells connected by 4-way
   grid neighbors (`STEP = 0.25`). This predates the Mapping Agent and is now
   effectively dead code — the live spatial graph is owned by
   `agents.mapping_agent.SpatialGraph`. Slated for removal.

5. **Plan generation.** `generate_plan(task, visual_input, top_down_view)`
   concatenates the system prompt, the task, the visual input, the map summary
   pulled from the Mapping Agent, and the top-down view, then sends the result
   to the LLM. The returned string is the proposed plan that would, in the
   full pipeline, be forwarded to the Critic Agent.

6. **Termination check (stub).** `should_terminate(task)` is a placeholder
   that, once wired up, will let the planner signal handoff back to the
   critic when the task is believed complete.

The `__main__` block sketches the entry point: it loads a house from
`prior`'s `procthor-10k` dataset and instantiates an AI2-THOR `Controller` on
it. Instantiating and exercising `PlanningAgent` from this script is the next
step.

## How this fits the week-5 framework

| `report/week5.tex` role         | Status                                                                       |
| ------------------------------- | ---------------------------------------------------------------------------- |
| Planning Agent (LLM core)       | Implemented in `plan/base.py` (`PlanningAgent.__call__` / `generate_plan`)   |
| Spatial Memory / Mapping Agent  | Implemented in `agents/mapping_agent.py` (`SpatialGraph` + `MappingAgent`)   |
| Perception Agent                | Implemented separately under `agents/perception`; feeds planner context, not required by Mapping Agent topology |
| Critic Agent                    | Not present — `should_terminate` is a stub for the handoff                   |
| Action Agent                    | Not present — plans are returned as text, not executed                       |

## Known gaps / TODOs

- `PlanningAgent.__call__` references `model="gpt-5.5"`, which is not a real
  model id and needs to be replaced with the chosen "thinking" LLM
  (DeepSeek / Claude / Qwen per the report).
- `PlanningAgent` does not inherit from `agents.base.AgentBase`. For
  consistency with the Mapping Agent (and the upcoming Critic Agent), it
  should be refactored onto the shared `BaseAgentCore` / `AgentBase` template.
- `create_graph` shadows its `reachable_positions` argument by re-calling
  `self.get_reachable_positions()`, never returns the constructed `G`, and is
  now superseded by `agents.mapping_agent.SpatialGraph`. Delete.
- `should_terminate` calls `self.task_terminate(task)`, but `task_terminate`
  is a string constant (`"Call to critic"`), not a callable. This needs to be
  replaced with an actual Critic Agent invocation.
- `visual_input` is passed to the LLM as a raw string; this should become a
  structured Perception Agent payload.
