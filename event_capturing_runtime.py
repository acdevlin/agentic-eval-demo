from builtins import str

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

    def _nested_tool_calls(self, execution_id, visited=None):
        visited = visited or set()
        if not execution_id or execution_id in visited:
            return []
        visited.add(execution_id)

        execution = self._rt._agent_client.get_execution(execution_id)
        tool_calls = []
        for task in execution.get("tasks", []):
            task_type = str(task.get("taskType", "")).upper()
            task_ref = str(task.get("referenceTaskName", ""))

            if "SUB_WORKFLOW" in task_type:
                tool_calls.extend(
                    self._nested_tool_calls(task.get("subWorkflowId"), visited)
                )
            elif task_ref.startswith("call_"):
                tool_calls.append(
                    {
                        "name": task_type.lower(),
                        "args": task.get("inputData", {}),
                        "result": task.get("outputData", {}),
                    }
                )
        return tool_calls