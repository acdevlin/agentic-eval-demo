from builtins import str
from types import SimpleNamespace
from conductor.ai.agents.runtime.runtime import _normalize_handoff_target


class EventCapturingRuntime:
    """
    Captures all events from an agentic run, including nested tool callchain.
    """

    def __init__(self, rt):
        self._rt = rt

    def run(self, agent, prompt):
        result = self._rt.stream(agent, prompt).get_result()
        result.tool_calls = self._nested_tool_calls(result.execution_id)
        return result

    def _nested_tool_calls(self, execution_id, agent=None, visited=None):
        if visited is None:
            visited = set()
        if not execution_id or execution_id in visited:
            return []
        visited.add(execution_id)

        try:
            wf = self._rt._workflow_client.get_workflow(
                execution_id, include_tasks=True
            )
        except Exception:
            return []  # match the SDK: enrichment is best-effort

        calls = []
        for task in getattr(wf, "tasks", None) or []:
            if "SUB_WORKFLOW" in str(getattr(task, "task_type", "")).upper():
                child = _normalize_handoff_target(
                    getattr(task, "reference_task_name", "")
                )
                calls.extend(
                    self._nested_tool_calls(task.sub_workflow_id, child, visited)
                )
            else:
                for call in self._rt._extract_tool_calls(SimpleNamespace(tasks=[task])):
                    calls.append({**call, "agent": agent, "execution_id": execution_id})
        return calls
