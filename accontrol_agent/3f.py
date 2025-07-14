import os
import json
import boto3
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
from io import BytesIO
from mangum import Mangum  # For AWS Lambda
import pytz

app = FastAPI()

# AWS S3 Configuration
S3_BUCKET_NAME = 'osakaminohc'
s3_client = boto3.client('s3')

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

position_data_cache = {}

# Authenticate and get the access token
async def APIAuth():
    url = 'https://api.hito-navi.net/api/v1/login/'
    username = 'handai.ichild2023@gmail.com'
    password = 'kA6EL5h7'

    body = {
        "username": username,
        "password": password
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=body)

    if response.status_code == 200:
        token_json = response.json()
        return token_json.get('access_token')
    else:
        print(f"Failed to authenticate. Status code: {response.status_code}, Response: {response.text}")
        raise HTTPException(status_code=response.status_code, detail="Failed to authenticate")

# def get_current_time_rounded():
#     jst = pytz.timezone('Asia/Tokyo')
#     now = datetime.now(jst) - timedelta(seconds=10)
#     return now.strftime("%Y-%m-%d+%H:%M:%S")

async def fetch_position_data():
    global position_data_cache
    print("Fetching position data...")  # Log job execution
    try:
        auth_token = await APIAuth()
        # start_time = get_current_time_rounded()
        position_url = f"https://api.hito-navi.net/api/v1/position/?area_id=10"
        headers = {"Authorization": f'Bearer {auth_token}'}

        print(f"Requesting data from URL: {position_url}")

        async with httpx.AsyncClient() as client:
            response = await client.get(position_url, headers=headers)

        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.text}")

        if response.status_code == 200:
            new_data = response.json()
            print(f"Fetched new data: {new_data}")  # Log fetched data

            if new_data and isinstance(new_data, list):
                if new_data[0]["time"] != (position_data_cache[0]["time"] if position_data_cache else None):
                    position_data_cache = new_data
                    print("Data received and stored in cache.")
                    store_position_data()
                    await store_position_image()
                else:
                    print("Data has not changed based on the time field.")
            else:
                print("API response is missing required data.")
        else:
            print(f"Failed to fetch data: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error occurred while fetching data: {str(e)}")

def store_position_data():
    global position_data_cache
    try:
        if position_data_cache:
            json_bytes = json.dumps(position_data_cache).encode('utf-8')
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key='position_data.json', Body=json_bytes, ContentType='application/json')
            print("Position data stored in S3.")
        else:
            print("Position data cache is empty. Skipping storage.")
    except Exception as e:
        print(f"Error storing position data in S3: {str(e)}")

async def store_position_image():
    try:
        await generate_position_image()  # Generate image first
    except Exception as e:
        print(f"Error storing position image in S3: {str(e)}")

async def generate_position_image():
    try:
        # Load the base image
        base_image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='base.webp')
        base_image_data = base_image_obj['Body'].read()
        img = Image.open(BytesIO(base_image_data)).convert("RGBA")
        
        # Load the overlay image
        overlay_image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='cover2.webp')
        overlay_image_data = overlay_image_obj['Body'].read()
        overlay_img = Image.open(BytesIO(overlay_image_data)).convert("RGBA")
    except Exception as e:
        print(f"Error loading images from S3: {str(e)}")
        return

    width, height = img.size
    draw = ImageDraw.Draw(img)

    x_scale = width / 81.3
    y_scale = height / 73.4
    x_offset = width * 0.5
    y_offset = height * 0.7

    # Draw basic points
    basic_points = [
        {"id": 1, "x": 34.32, "y": -23.25},
        {"id": 2, "x": -17.7, "y": -23.25},
        {"id": 3, "x": 34.32, "y": 2.95},
        {"id": 4, "x": 0.0, "y": 0.0},
        {"id": 5, "x": -17.7, "y": 2.95}
    ]
    # for point in basic_points:
    #     draw_position(draw, point, x_scale, y_scale, x_offset, y_offset)

    # Draw basic points regardless of num
    if position_data_cache:
        for entry in position_data_cache:
            if entry.get("num", 0) > 5:
                if "data" in entry and entry["data"] is not None:
                    for human in entry["data"]:
                        draw_position(draw, human, x_scale, y_scale, x_offset, y_offset)

    # Overlay the overlay image
    img.paste(overlay_img, (0, 0), overlay_img)

    try:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key='3F_human_position.png', Body=img_byte_arr, ContentType='image/png')
        print("Annotated image with overlay successfully saved to S3.")
    except Exception as e:
        print(f"Error saving image to S3: {str(e)}")


def draw_position(draw, human, x_scale, y_scale, x_offset, y_offset):
    
    base_image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='base.webp')
    base_image_data = base_image_obj['Body'].read()
    img = Image.open(BytesIO(base_image_data)).convert("RGBA")
    
    width,height = img.size
    # print(height)
    x = (-human["y"] * x_scale) + x_offset
    y = -((human["x"] * y_scale) + y_offset)+height
    color = "red"

    if human["id"] == 4:
        color = "orange"
    elif human["id"] in [1, 2, 3, 5]:
        color = "blue"

    ellipse_radius = 10
    draw.ellipse((x - ellipse_radius, y - ellipse_radius, x + ellipse_radius, y + ellipse_radius), fill=color)
    # draw.text((x + 35, y - 10), f'ID: {human["id"]}', fill="green")


@app.get("/api/positions")
async def get_position_data():
    try:
        if not position_data_cache:
            print("Position data cache is empty. Fetching new data.")
            await fetch_position_data()
        else:
            print("Returning cached position data.")
        return JSONResponse(content=position_data_cache)
    except Exception as e:
        print(f"Error in /api/positions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get('/api/human')
async def get_position_image():
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='3F_human_position.png')
        img_data = response['Body'].read()
        print(f"Image size: {len(img_data)} bytes")
        return StreamingResponse(BytesIO(img_data), media_type='image/png')
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Annotated image not found in S3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image from S3: {str(e)}")

@app.on_event("startup")
async def startup_event():
    # Preload position data on startup
    try:
        await fetch_position_data()
        print("Initial data fetch completed on startup.")
    except Exception as e:
        print(f"Error occurred during startup: {str(e)}")

mangum_handler = Mangum(app)

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if "httpMethod" in event:
        print("Handling API Gateway event.")
        return mangum_handler(event, context)
    else:
        print("Handling direct invocation event.")
        return direct_lambda_handler(event, context)

def direct_lambda_handler(event, context):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(fetch_position_data())
        else:
            asyncio.run(fetch_position_data())
        return {
            "statusCode": 200,
            "body": "Data fetched and saved to S3 successfully."
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error occurred: {str(e)}"
        }
