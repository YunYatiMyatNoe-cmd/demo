from typing import TypedDict, Optional

class AgentState(TypedDict):
    user_input: str
    processed_input: Optional[str]
    room: Optional[str]
    device_id: Optional[str]
    tool_results: Optional[str]
    knowledge_base_results: Optional[str]
    orchistrator_response: Optional[str]
    final_result: Optional[str]
    validation_result: Optional[str]
    validation_passed: bool
    retry_count: int
    error: Optional[str]
    next_action: Optional[str]
