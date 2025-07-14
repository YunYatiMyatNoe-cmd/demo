import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime

def get_weather_data():
    """Get current weather data for Minoh Campus, Osaka, Japan."""
    try:
        print("Fetching weather data...")

        # Setup Open-Meteo API client with caching and retries
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # API endpoint and parameters
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

        # Make API call
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        current = response.Current()

        # Convert time (in Unix timestamp) to human-readable JST
        timestamp = current.Time()
        time_jst = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Extract weather data
        data = {
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

        return data

    except Exception as e:
        return {"error": str(e)}

# Run and print result for testing
if __name__ == "__main__":
    weather = get_weather_data()
    print("\nCurrent Weather at Minoh Campus:")
    for key, value in weather.items():
        print(f"{key}: {value}")
