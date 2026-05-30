from __future__ import annotations

from typing import Any

import httpx

from ai_sdk_stream_python import StreamContext


async def handle_get_weather(
    input: dict[str, Any],
    ctx: StreamContext,
) -> dict[str, Any]:
    latitude: float | None = input.get("latitude")
    longitude: float | None = input.get("longitude")
    city: str | None = input.get("city")

    if city:
        async with httpx.AsyncClient() as client:
            geo_resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10.0,
            )
        if geo_resp.status_code != 200:
            return {"error": f'Could not geocode city "{city}"'}
        geo_data = geo_resp.json()
        results = geo_data.get("results") or []
        if not results:
            return {
                "error": f'Could not find coordinates for "{city}". Please check the city name.'
            }
        latitude = results[0]["latitude"]
        longitude = results[0]["longitude"]
    elif latitude is None or longitude is None:
        return {
            "error": "Please provide either a city name or both latitude and longitude coordinates."
        }

    async with httpx.AsyncClient() as client:
        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m",
                "hourly": "temperature_2m",
                "daily": "sunrise,sunset",
                "timezone": "auto",
            },
            timeout=10.0,
        )

    weather_data = weather_resp.json()
    if city:
        weather_data["cityName"] = city
    return weather_data
