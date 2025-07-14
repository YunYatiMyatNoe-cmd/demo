from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from accontrol_agent.utils.state import AgentState
from accontrol_agent.utils.nodes import (
    interface_agent, orchestrator_agent, validation_agent,
    should_retry
)

def create_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("interface_agent", interface_agent)
    graph.add_node("orchestrator_agent", orchestrator_agent)
    graph.add_node("validation_agent", validation_agent)

    graph.set_entry_point("interface_agent")
    graph.add_conditional_edges(
        "interface_agent",
        lambda state: "orchestrator_agent" if state.get("next_action") != "end" else "end",
        {
            "orchestrator_agent": "orchestrator_agent",
            "end": END
        }
    )

    graph.add_edge("orchestrator_agent", "validation_agent")
    
    graph.add_conditional_edges(
        "validation_agent",
        should_retry,
        {
            "retry_orchestrator": "orchestrator_agent",
            "format_output": "interface_agent",    
        }
    )

    return graph.compile()
