import re
import os
import boto3
import json
from dotenv import load_dotenv
from supabase import create_client
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AWS_REGION = "ap-northeast-1"
KNOWLEDGE_BASE_ID = "JOSLJLSFLZ"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
kb_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    client=bedrock_client,
)

ROOM_ALIASES = {
    "salc": "404 SALC",
    "cw2": "403 CW2",
    "halc": "409 HALC",
    "piloty": "ピロティ",
    "lounge": "交流スペース西",
    "食堂": "食堂",
    "server room": "サーバールーム２",
    "entrance": "風除室",
}

@tool
def get_room_data(room: str):
    """Get sensor data for a specific room."""
    try:
        parsed = json.loads(room)
        room_name = parsed["room"]
        print(f"Fetching data for room: {room}")
        print(f"Fetching data for room name: {room_name}")
        data = supabase.rpc('get_room_anomaly', {'room_input': room_name}).execute()
        print(f"Retrieved data for room {room_name}: {data.data}")
        return data.data if data.data else {"error": f"No data found for room: {room_name}"}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_device_data(device_id: str):
    """Get sensor data for a specific device by ID."""
    try:
        parsed = json.loads(device_id)
        device_id_name = parsed["device_id"]
        data = supabase.rpc('get_device_anomaly', {'device_input': device_id_name}).execute()
        print(f"Retrieved data for room {device_id_name}: {data.data}")
        return data.data if data.data else {"error": f"No data found for device: {device_id}"}
    except Exception as e:
        return {"error": str(e)}


def search_knowledge_base(query: str, tool_results: list[dict], advice: str = "") -> str:
    prompt = f"""以下の質問と、与えられる情報に基づき関連する文章を抽出してください。
    質問：{query}
    情報：{tool_results}
    アドバイス：{advice}
    """
    try:
        response = kb_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": prompt}
        )
        results = response['retrievalResults'][:5] if response['retrievalResults'] else []
        extracted_results = []
        for item in results:
            content = item.get("content", {}).get("text", "")
            document_uri = item.get("location", {}).get("s3Location", {}).get("uri", "")
            score = item.get("score", 0)
            extracted_results.append({
                'Content': content,
                'DocumentURI': document_uri,
                'Score': score
            })
        return f"参考マニュアル情報：\n{str(extracted_results)}"
    except Exception as e:
        return f"Error retrieving from knowledge base: {str(e)}"

def extract_room_name(text: str) -> str | None:
    text_lower = text.lower()
    for alias, official in ROOM_ALIASES.items():
        if alias in text_lower:
            return official
    return None

def extract_device_id(text: str) -> str | None:
    pattern = r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
    match = re.search(pattern, text)
    return match.group(0) if match else None

def run_interface(prompt: str) -> str:
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    try:
        input_data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31"
        }
        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
            body=json.dumps(input_data),
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body.get('content', [{}])[0].get('text', 'No response generated')
    except Exception as e:
        return f"Error querying Bedrock: {str(e)}"
