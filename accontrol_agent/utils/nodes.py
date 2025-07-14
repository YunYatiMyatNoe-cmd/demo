import json
import re
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from accontrol_agent.utils.tools import (
    get_room_data, get_device_data,get_weather_data,search_knowledge_base,
    extract_room_name, extract_device_id, llm, run_interface
)
from accontrol_agent.utils.state import AgentState

def interface_agent(state: AgentState) -> AgentState:
    """
    Interface Agent - Input/Output Management
    Handles input processing, entity extraction, and output formatting.
    Always ensures structured output format.
    """
    user_input = state.get("user_input", "").strip()
    previous_input = state.get("processed_input")
    next_action = state.get("next_action")
    final_result = state.get("final_result")
    print("Final Result", final_result)

    # Handle end / output step
    if next_action in ("format_output", "end"):
        if user_input and user_input != previous_input:
            print("[interface_agent] New input detected after end. Restarting orchestration.")
            state.update({
                "processed_input": user_input,
                "room": extract_room_name(user_input),
                "device_id": extract_device_id(user_input),
                "retry_count": 0,
                "validation_passed": False,
                "validation_result": "",
                "error": None,
                "final_result": None,
                "tool_results": None,
                "knowledge_base_results": None,
                "orchistrator_response": None,
                "improved_result": None,
                "output": None,
                "next_action": "orchestrator_agent"
            })
            return state

        #Structure output format
        structured_output = {
            "answer": final_result or "No result generated.",
            "room": state.get("room"),
            "device_id": state.get("device_id"),
            "error": state.get("error"),
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }

        state["output"] = {"text": structured_output}
        state["next_action"] = "end"
        print(f"[interface_agent] Final structured output: {structured_output}")
        return state

    # Handle new input
    if user_input:
        print(f"[interface_agent] Received input: '{user_input}' (previous: '{previous_input}')")
        state.update({
            "processed_input": user_input,
            "room": extract_room_name(user_input),
            "device_id": extract_device_id(user_input),
            "retry_count": 0,
            "validation_passed": False,
            "validation_result": "",
            "error": None,
            "final_result": None,
            "tool_results": None,
            "knowledge_base_results": None,
            "orchistrator_response": None,
            "improved_result": None,
            "output": None,
            "next_action": "orchestrator_agent"
        })
        print("[interface_agent] State initialized for orchestrator_agent step.")

    # Always guarantee an output structure
    if not state.get("output"):
        state["output"] = {
            "text": {
                "answer": None,
                "room": state.get("room"),
                "device_id": state.get("device_id"),
                "error": state.get("error"),
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
        }
    return state
def orchestrator_agent(state: AgentState) -> AgentState:
    """
    Orchestrator Agent - Task Decomposition and Control
    Handles task decomposition, tool selection, and data collection
    """
    user_input = state["processed_input"]
    room = state.get("room")
    device_id = state.get("device_id")
    
    # Initialize LLM with tools
    llm_with_tools = llm.bind_tools([get_room_data, get_device_data, get_weather_data])
    
    system_prompt = f"""You are a smart building orchestrator agent responsible for task decomposition and control.
    
    Your role:
    1. Analyze the user's query and determine what data is needed
    2. Use appropriate tools to collect sensor data
    3. Coordinate with knowledge base for additional context
    
    Current context:
    - Room: {room}
    - Device: {device_id}
    
    Based on the user's input, determine what data to collect and use the appropriate tools."""

    try:
        print("Reach Orchestrator Agent")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_input}")
        ])
        chain = prompt | llm_with_tools
        result = chain.invoke({"user_input": user_input})

        # Process tool calls and collect data
        tool_results = []
        
        if isinstance(result.content, list):
            for item in result.content:
                if item.get("type") == "tool_use":
                    if item.get("name") == "get_room_data":
                        # print("Name",item.get("name"))
                        # print(f"............Invoking get_room_data with input: {item['input']}")
                        tool_result = get_room_data.invoke(item["input"])
                        tool_results.append(tool_result)
                    elif item.get("name") == "get_device_data":
                        # print(f"............Invoking get_device_data with input: {item['input']}")
                        tool_result = get_device_data.invoke(item["input"])
                        tool_results.append(tool_result)
                    elif item.get("name") == "get_weather_data":
                        # print(f"............Invoking get_room_data with input: {item['input']}")
                        tool_result = get_weather_data.invoke(item["input"])
                        tool_results.append(tool_result)

        # Get knowledge base information
        knowledge_base_results = ""
        advice = state.get("improved_result", "")
        knowledge_base_results = search_knowledge_base(user_input, tool_results, advice)
        # print(f"Knowledge Base Results: {knowledge_base_results}")
        
        # Generate initial response
        if tool_results or knowledge_base_results:
            response_prompt = f"""
            Based on the collected data, Please provide the best possible answer based on the user's question."
            
            User question：{user_input}
            Available data：
            {str(tool_results)}
            {knowledge_base_results}
            Advice：{advice}
        """           
            orchistrator_response = run_interface(response_prompt)
            print(f"Initial Response: {orchistrator_response}")
        
        state.update({
            "tool_results": str(tool_results),
            "knowledge_base_results": knowledge_base_results,
            "orchistrator_response": orchistrator_response
        })
        
        # print(f"Orchestrator Agent - Collected data, generated initial response")
        
    except Exception as e:
        state["error"] = str(e)
        state["orchistrator_response"] = f"データ収集中にエラーが発生しました: {str(e)}"
        print(f"Orchestrator Agent Error: {str(e)}")
    
    return state

def validation_agent(state: AgentState) -> AgentState:
    """
    Validation Agent (Quality Assurance)
    Validates response quality and determines if retry is needed (max 3 retries based on score)
    """
    orchistrator_response = state.get("orchistrator_response", "")
    user_input = state.get("processed_input", "")
    
    print("Reach Validation Agent")
    
    validation_prompt = f"""You are a Quality Assurance expert. Please evaluate 
    the following response strictly based on the criteria below:

    Evaluation Criteria:
    1. Relevance: Is the response directly related to the user's question? (score 0-100%)
    2. Completeness: Does the response provide sufficient information to address the question? (score 0-100%)
    3. Accuracy: Is the response factually correct based on the provided data? (score 0-100%)
    4. Consistency: Is the response free from contradictions or internal inconsistencies? (score 0-100%)

    User's Question: {user_input}
    Generated Response: {orchistrator_response}

    Please provide your strict evaluation in the following JSON format:
    {{
    "relevance": {{"score": int, "reason": str}},
    "completeness": {{"score": int, "reason": str}},
    "accuracy": {{"score": int, "reason": str}},
    "consistency": {{"score": int, "reason": str}},
    "improved_response": "Optional improved response if applicable",
    }}
    """
    try:
        validation_result_raw = run_interface(validation_prompt)
        # print("Validation Result", validation_result_raw)

        try:
            validation_result = json.loads(validation_result_raw)
        except Exception:
            match = re.search(r"\{.*\}", validation_result_raw, re.DOTALL)
            if match:
                validation_result = json.loads(match.group(0))
            else:
                raise ValueError("Could not parse LLM output as JSON.")
            
        print("Parsed Validation Result", validation_result)

        validation_passed = (
            validation_result.get("relevance", {}).get("score", 0) >= 80 and
            validation_result.get("completeness", {}).get("score", 0) >= 80 and
            validation_result.get("accuracy", {}).get("score", 0) >= 80 and
            validation_result.get("consistency", {}).get("score", 0) >= 80
        )

        if validation_passed:
            state.update({
                "final_result": orchistrator_response,
                "validation_result": validation_result,
                "next_action": "format_output",
                "validation_passed": True,
            })
        else:
            retry_count = state.get("retry_count", 0)
            if retry_count < 3:
                state.update({
                    "improved_result": validation_result.get("improved_response", ""),
                    "retry_count": retry_count + 1,
                    "validation_result": validation_result,
                    "next_action": "retry_orchestrator",
                    "validation_passed": False,
                })
            else:
                state.update({
                    "final_result": orchistrator_response,
                    "validation_result": validation_result,
                    "validation_passed": False,
                    "next_action": "format_output",
                    "error": "Validation failed after maximum retries"
                })

    except Exception as e:
        state["error"] = str(e)
        state["final_result"] = orchistrator_response
        state["validation_passed"] = False
        state["next_action"] = "format_output"
        print(f"Validation Agent Error: {str(e)}")
    
    return state

def should_retry(state: AgentState) -> str:
    return "retry_orchestrator" if state.get("next_action") == "retry_orchestrator" else "format_output"


