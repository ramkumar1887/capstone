"""
Weather service for the remedy module.
Uses Open-Meteo (free, no API key). Geocodes a place name, fetches current
conditions + 3-day forecast, and turns them into farm-relevant advisories.

All network calls fail soft: if weather is unavailable, the remedy module
still works and simply omits weather advice.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10

# Disease categories drive which weather rules apply
DISEASE_TYPE = {
    "Phytophthora Blight": "fungal",
    "Fruit rot": "fungal",
    "Cercospora Leaf Spot": "fungal",
    "Phomopsis blight": "fungal",
    "Wilt Disease": "soilborne",
    "Fruit borer": "insect",
    "Melon thrips": "insect",
    "Tobacco Mosaic Virus": "viral",
}


def canonical_disease(name):
    """Map a model class name (e.g. 'Fruit_borer') to the canonical name used in the
    rule table, ChromaDB metadata and weather logic (e.g. 'Fruit borer').
    Matching ignores case, underscores, hyphens and extra spaces."""
    def key(s):
        return " ".join(s.replace("_", " ").replace("-", " ").lower().split())
    lookup = {key(k): k for k in DISEASE_TYPE}
    return lookup.get(key(name), name)


def geocode(place):
    """Return (lat, lon, display_name) or None."""
    try:
        r = requests.get(GEOCODE_URL, params={"name": place, "count": 1}, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results")
        if not results:
            return None
        top = results[0]
        name = ", ".join(x for x in [top.get("name"), top.get("admin1"), top.get("country")] if x)
        return top["latitude"], top["longitude"], name
    except Exception:
        return None


def fetch_weather(place):
    """Return a normalized weather dict, or None if unavailable."""
    loc = geocode(place)
    if loc is None:
        return None
    lat, lon, display = loc
    try:
        r = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        cur, daily = data["current"], data["daily"]
        return {
            "location": display,
            "temp_c": cur["temperature_2m"],
            "humidity": cur["relative_humidity_2m"],
            "wind_kmh": cur["wind_speed_10m"],
            "rain_now_mm": cur["precipitation"],
            "rain_prob_today": daily["precipitation_probability_max"][0],
            "rain_prob_next_48h": max(daily["precipitation_probability_max"][:2]),
            "rain_mm_next_48h": sum(daily["precipitation_sum"][:2]),
            "temp_max_today": daily["temperature_2m_max"][0],
        }
    except Exception:
        return None


def weather_advisories(disease, weather):
    """
    Turn weather + disease into a dict:
      urgency: "Normal" | "Elevated"
      risk_notes: list of why conditions help or hinder the disease
      spray_advice: list of application timing/safety notes
    Rules are deliberately general agronomic guidance, not product-specific.
    """
    if weather is None:
        return None

    dtype = DISEASE_TYPE.get(disease, "fungal")
    risk_notes, spray_advice = [], []
    urgency = "Normal"

    humid = weather["humidity"] >= 80
    wet = weather["rain_prob_next_48h"] >= 60 or weather["rain_mm_next_48h"] >= 5
    warm = 20 <= weather["temp_c"] <= 32
    hot_dry = weather["temp_c"] >= 30 and weather["humidity"] <= 55

    # --- Disease-risk notes ---
    if dtype == "fungal":
        if humid and (wet or warm):
            urgency = "Elevated"
            risk_notes.append("High humidity with warm/wet conditions favours fungal spread. "
                              "Act promptly and prioritise removing infected material.")
        elif weather["humidity"] < 60:
            risk_notes.append("Dry conditions slow fungal spread, but keep monitoring.")
    elif dtype == "soilborne":
        if wet:
            urgency = "Elevated"
            risk_notes.append("Wet soil conditions favour soil-borne wilt pathogens. "
                              "Improve drainage and avoid irrigating until rain passes.")
    elif dtype == "insect":
        if disease == "Melon thrips" and hot_dry:
            urgency = "Elevated"
            risk_notes.append("Hot, dry weather favours thrips build-up. Increase monitoring.")
        elif disease == "Fruit borer" and humid and warm:
            urgency = "Elevated"
            risk_notes.append("Warm, humid weather favours fruit borer activity. "
                              "Check shoots and fruit frequently.")
    elif dtype == "viral":
        risk_notes.append("Virus spreads via handling and insect vectors, not rain. "
                          "Weather does not change the core actions: remove infected plants "
                          "and disinfect tools.")

    # --- Spray timing / safety (not relevant to viral: nothing to spray) ---
    if dtype != "viral":
        if weather["rain_prob_today"] >= 60 or weather["rain_now_mm"] > 0:
            spray_advice.append("Rain is likely today: delay spraying, as rain washes sprays off. "
                                "Wait for a dry spell of at least 24 hours.")
        elif weather["rain_prob_next_48h"] >= 60:
            spray_advice.append("Rain is likely within 48 hours: spray only if you can do it "
                                "early today, otherwise wait until after the rain.")
        if weather["wind_kmh"] >= 15:
            spray_advice.append(f"Wind is {weather['wind_kmh']:.0f} km/h: avoid spraying "
                                "to prevent drift; wait for calmer conditions.")
        if weather["temp_max_today"] >= 35:
            spray_advice.append("Very hot day: spray early morning or late evening to avoid "
                                "leaf burn and fast evaporation.")
        if not spray_advice:
            spray_advice.append("Conditions look suitable for spraying (low rain chance, light "
                                "wind). Early morning or evening is still best.")

    return {"urgency": urgency, "risk_notes": risk_notes, "spray_advice": spray_advice}
