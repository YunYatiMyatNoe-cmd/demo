import re
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
import os
# os.environ["REQUESTS_CA_BUNDLE"] = ""
import boto3
import json
import openmeteo_requests
import requests_cache
from retry_requests import retry
from dotenv import load_dotenv
from datetime import datetime, timedelta
from supabase import create_client
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AWS_REGION = "us-east-1"
KNOWLEDGE_BASE_ID = "IZXJ8417SA"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
kb_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

llm = ChatBedrockConverse(
    model=model_id,
    temperature=0,
    client=bedrock_client,
)

ROOM_ALIASES = {
    "402 CW1": "402 CW1",
    "交流スペース（6F）": "交流スペース（6F）",
    "交流ラウンジ": "交流ラウンジ",
    "403 CW2": "403 CW2",
    "405 中講義室": "405 中講義室",
    "風除室": "風除室",
    "404 SALC": "404 SALC",
    "交流スペース東": "交流スペース東",
    "サーバールーム２": "サーバールーム２",
    "階段教室": "階段教室",
    "交流スペース（4F）": "交流スペース（4F）",
    "交流スペース西": "交流スペース西",
    "ピロティ": "ピロティ",
    "学生交流スペース": "学生交流スペース",
    "409 HALC": "409 HALC",
    "食堂": "食堂"
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
    

@tool
def get_weather_data():
    """Get current weather data for Minoh Campus, Osaka, Japan. can use to comfort air control system."""
    try:
        # Setup Open-Meteo API client with caching and retries
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 34.664967,
            "longitude": 135.451014,
            "current": [
                "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "wind_direction_10m", "apparent_temperature", "is_day",
                "wind_gusts_10m", "precipitation", "rain", "showers",
                "snowfall", "weather_code", "cloud_cover",
                "pressure_msl", "surface_pressure"
            ],
            "timezone": "Asia/Tokyo"
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        current = response.Current()
        print("Current data object:", current)
        print("Available current methods:", dir(current))

        # Convert UNIX timestamp to human-readable JST
        timestamp = current.Time()
        time_jst = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Extract all requested variables
        return {
            "time_jst": time_jst,
            "temperature_C": current.Variables(0).Value(),
            "relative_humidity_%": current.Variables(1).Value(),
            "wind_speed_10m_kmh": current.Variables(2).Value(),
            "wind_direction_10m_deg": current.Variables(3).Value(),
            "apparent_temperature_C": current.Variables(4).Value(),
            "is_day": current.Variables(5).Value(),
            "wind_gusts_10m_kmh": current.Variables(6).Value(),
            "precipitation_mm": current.Variables(7).Value(),
            "rain_mm": current.Variables(8).Value(),
            "showers_mm": current.Variables(9).Value(),
            "snowfall_mm": current.Variables(10).Value(),
            "weather_code": current.Variables(11).Value(),
            "cloud_cover_%": current.Variables(12).Value(),
            "pressure_msl_hPa": current.Variables(13).Value(),
            "surface_pressure_hPa": current.Variables(14).Value()
        }

    except Exception as e:
        import traceback
        print(f"Error fetching weather data: {str(e)}")
        traceback.print_exc()
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
            "thinking": {"type": "enabled", "budget_tokens": 1600},
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31"
        }
        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
            body=json.dumps(input_data),
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        # print("----------response_body----------", response_body)
        text_content = next(
        (item.get("text") for item in response_body.get("content", []) if item.get("type") == "text"),
        "No response generated"
    )
        print("----------orchestrator response_body----------", text_content)
        return text_content
    except Exception as e:
        print(f"Error querying Bedrock: {str(e)}")
        return f"Error querying Bedrock: {str(e)}"