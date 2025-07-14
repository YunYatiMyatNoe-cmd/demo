from typing import TypedDict, Optional,Dict, Any, List

class AgentState(TypedDict):
    user_input: str
    processed_input: Optional[str]
    room: Optional[str]
    device_id: Optional[str]
    tool_results: Optional[List[Any]]
    knowledge_base_results: Optional[str]
    orchistrator_response: Optional[str]
    final_result: Optional[str]
    validation_result: Optional[Dict[str, Any]]
    validation_passed: bool
    retry_count: int
    error: Optional[str]
    next_action: Optional[str]
    output: Optional[Dict[str, Any]] 
