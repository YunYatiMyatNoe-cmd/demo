import json
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
    Handles input processing, entity extraction, and output formatting
    """
    user_input = state.get("user_input", "").strip()
    room = extract_room_name(user_input)
    device_id = extract_device_id(user_input)
    print(f"[interface_agent] extracted room={room}, device_id={device_id}")

    # Ensure all fields initialized so graph input can be minimal
    state.setdefault("processed_input", user_input)
    state.setdefault("retry_count", 0)
    state.setdefault("validation_passed", False)
    state.setdefault("validation_result", "")
    state.setdefault("error", None)
    state.update({
        "processed_input": user_input,
        "room": room,
        "device_id": device_id
    })
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
                        print("Name",item.get("name"))
                        print(f"............Invoking get_room_data with input: {item['input']}")
                        tool_result = get_room_data.invoke(item["input"])
                        tool_results.append(tool_result)
                    elif item.get("name") == "get_device_data":
                        print(f"............Invoking get_device_data with input: {item['input']}")
                        tool_result = get_device_data.invoke(item["input"])
                        tool_results.append(tool_result)
                    elif item.get("name") == "get_weather_data":
                        print(f"............Invoking get_room_data with input: {item['input']}")
                        tool_result = get_weather_data.invoke(item["input"])
                        tool_results.append(tool_result)

        # Get knowledge base information
        knowledge_base_results = ""
        advice = state.get("improved_result", "")
        knowledge_base_results = search_knowledge_base(user_input, tool_results, advice)
        print(f"Knowledge Base Results: {knowledge_base_results}")
        
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
        
        print(f"Orchestrator Agent - Collected data, generated initial response")
        
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
    tool_results = state.get("tool_results", "")
    knowledge_base_results = state.get("knowledge_base_results", "")
    
    print("Validation Agent - Validating response quality")
    
    validation_prompt = f"""You are a Quality Assurance expert. Please evaluate 
    the following response strictly based on the criteria below:

    Evaluation Criteria:
    1. Relevance: Is the response directly related to the user's question? (score 0-100%)
    2. Completeness: Does the response provide sufficient information to address the question? (score 0-100%)
    3. Accuracy: Is the response factually correct based on the provided data? (score 0-100%)
    4. Consistency: Is the response free from contradictions or internal inconsistencies? (score 0-100%)

    User's Question: {user_input}
    Generated Response: {orchistrator_response}
    Available Tool Results: {tool_results}
    Available Knowledge Base: {knowledge_base_results}

    Please provide your strict evaluation in the following JSON format:
    {{
    "relevance": {{"score": int, "reason": str}},
    "completeness": {{"score": int, "reason": str}},
    "accuracy": {{"score": int, "reason": str}},
    "consistency": {{"score": int, "reason": str}},
    "average": int,
    "judgment": "PASS" or "FAIL",
    "improved_response": str or null,
    "final_response": str
    }}
"""
    try:
        validation_result_raw = run_interface(validation_prompt)
        print("Validation Result", validation_result_raw)

        try:
            validation_result = json.loads(validation_result_raw)
        except Exception:
            # Fallback: try to extract JSON from text
            import re
            match = re.search(r"\{.*\}", validation_result_raw, re.DOTALL)
            if match:
                validation_result = json.loads(match.group(0))
            else:
                raise ValueError("Could not parse LLM output as JSON.")

        validation_passed = validation_result.get("judgment", "").upper() == "PASS"
        retry_count = state.get("retry_count", 0)

        if validation_passed or retry_count >= 4:
            state.update({
                "final_result": validation_result.get("final_response", orchistrator_response),
                "validation_passed": validation_passed,
                "validation_result": validation_result,
                "next_action": "end"
            })
            if not validation_passed:
                print("Validation Agent - MAX RETRIES REACHED")
            else:
                print("Validation Agent - PASSED")
        else:
            state.update({
                "improved_result": validation_result.get("improved_response", ""),
                "validation_passed": False,
                "retry_count": retry_count + 1,
                "validation_result": validation_result,
                "next_action": "retry_orchestrator"
            })
            print("Validation Agent - IMPROVED")
        
        
    except Exception as e:
        state["error"] = str(e)
        state["final_result"] = orchistrator_response  # Fallback to initial response
        state["validation_passed"] = False
        state["next_action"] = "end"
        print(f"Validation Agent Error: {str(e)}")
    
    return state

def should_retry(state: AgentState) -> str:
    return "retry_orchestrator" if state.get("next_action") == "retry_orchestrator" else "format_output"


