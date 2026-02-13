"""
CAM Highway 20 Documentary Planner

Production planning tools for the US Route 20 documentary project (June 2026).
Newport, OR to Boston, MA — ~3,365 miles, the longest US road.

Manages segment planning, shot lists, progress tracking, weather planning,
and scouting notes. SQLite-backed with async on_change callback for
dashboard integration.

Usage:
    from tools.content.highway20 import Highway20Planner

    planner = Highway20Planner(
        db_path="data/highway20.db",
        router=model_router,
        ltm=long_term_memory,
        on_change=broadcast_hwy20_status,
    )
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger("cam.highway20")

# ---------------------------------------------------------------------------
# Constants — US Route 20 states in west-to-east order
# ---------------------------------------------------------------------------

HWY20_STATES = ["OR", "ID", "MT", "WY", "NE", "IA", "IL", "IN", "OH", "PA", "NY", "MA"]

TERRAIN_TYPES = [
    "coastal", "mountain", "forest", "desert", "plains",
    "farmland", "urban", "suburban", "river_valley", "lake",
]

# ---------------------------------------------------------------------------
# Static Seasonal Weather Data (zero API cost, works offline)
# ---------------------------------------------------------------------------

# Each entry: {spring, summer, fall, winter} with avg_temp_f, rain_chance_pct,
# wind_mph, filming_rating (1-5), notes
SEASONAL_WEATHER: dict[tuple[str, str], dict] = {
    # Oregon — coastal, mountain, forest
    ("OR", "coastal"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 55, "wind_mph": 15, "filming_rating": 3, "notes": "Fog common mornings, clears by noon"},
        "summer": {"avg_temp_f": 62, "rain_chance_pct": 15, "wind_mph": 12, "filming_rating": 5, "notes": "Best filming — clear, long days"},
        "fall":   {"avg_temp_f": 54, "rain_chance_pct": 50, "wind_mph": 14, "filming_rating": 3, "notes": "Beautiful colors but rain returns"},
        "winter": {"avg_temp_f": 45, "rain_chance_pct": 70, "wind_mph": 18, "filming_rating": 1, "notes": "Heavy rain, short days"},
    },
    ("OR", "mountain"): {
        "spring": {"avg_temp_f": 45, "rain_chance_pct": 40, "wind_mph": 10, "filming_rating": 3, "notes": "Snow possible at passes, wildflowers lower"},
        "summer": {"avg_temp_f": 68, "rain_chance_pct": 10, "wind_mph": 8, "filming_rating": 5, "notes": "Clear skies, excellent visibility"},
        "fall":   {"avg_temp_f": 48, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 4, "notes": "Fall colors in Cascades"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 60, "wind_mph": 15, "filming_rating": 1, "notes": "Snow, chain requirements, passes may close"},
    },
    ("OR", "forest"): {
        "spring": {"avg_temp_f": 50, "rain_chance_pct": 45, "wind_mph": 8, "filming_rating": 3, "notes": "Green canopy emerging, muddy trails"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 10, "wind_mph": 6, "filming_rating": 5, "notes": "Dry, warm, great light through canopy"},
        "fall":   {"avg_temp_f": 52, "rain_chance_pct": 40, "wind_mph": 8, "filming_rating": 4, "notes": "Fall colors, mushroom season"},
        "winter": {"avg_temp_f": 38, "rain_chance_pct": 65, "wind_mph": 10, "filming_rating": 2, "notes": "Wet, low light, limited access"},
    },
    # Idaho — mountain, desert, river_valley
    ("ID", "mountain"): {
        "spring": {"avg_temp_f": 42, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 3, "notes": "Late snow, muddy roads"},
        "summer": {"avg_temp_f": 70, "rain_chance_pct": 15, "wind_mph": 8, "filming_rating": 5, "notes": "Clear, dry, stunning scenery"},
        "fall":   {"avg_temp_f": 45, "rain_chance_pct": 25, "wind_mph": 10, "filming_rating": 4, "notes": "Golden aspens, cool temps"},
        "winter": {"avg_temp_f": 22, "rain_chance_pct": 50, "wind_mph": 15, "filming_rating": 1, "notes": "Heavy snow, road closures likely"},
    },
    ("ID", "desert"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 15, "filming_rating": 4, "notes": "Wildflowers in high desert"},
        "summer": {"avg_temp_f": 88, "rain_chance_pct": 8, "wind_mph": 10, "filming_rating": 3, "notes": "Hot, harsh midday light"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 15, "wind_mph": 12, "filming_rating": 5, "notes": "Perfect temps, golden light"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 30, "wind_mph": 15, "filming_rating": 2, "notes": "Cold, possible black ice"},
    },
    ("ID", "river_valley"): {
        "spring": {"avg_temp_f": 50, "rain_chance_pct": 30, "wind_mph": 10, "filming_rating": 4, "notes": "Green valleys, snowmelt rivers"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 10, "wind_mph": 8, "filming_rating": 5, "notes": "Warm, clear, great reflections"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 4, "notes": "Cottonwood gold along rivers"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 40, "wind_mph": 12, "filming_rating": 2, "notes": "Valley fog, cold"},
    },
    # Montana — mountain, plains
    ("MT", "mountain"): {
        "spring": {"avg_temp_f": 40, "rain_chance_pct": 35, "wind_mph": 15, "filming_rating": 3, "notes": "Late season snow, dramatic skies"},
        "summer": {"avg_temp_f": 68, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "Big Sky country at its best"},
        "fall":   {"avg_temp_f": 42, "rain_chance_pct": 20, "wind_mph": 12, "filming_rating": 4, "notes": "Larch gold, elk bugling"},
        "winter": {"avg_temp_f": 18, "rain_chance_pct": 45, "wind_mph": 20, "filming_rating": 1, "notes": "Severe cold, road closures"},
    },
    ("MT", "plains"): {
        "spring": {"avg_temp_f": 48, "rain_chance_pct": 30, "wind_mph": 18, "filming_rating": 3, "notes": "Windy, greening up"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 20, "wind_mph": 12, "filming_rating": 5, "notes": "Long days, dramatic clouds"},
        "fall":   {"avg_temp_f": 48, "rain_chance_pct": 15, "wind_mph": 15, "filming_rating": 4, "notes": "Harvest colors, cooler"},
        "winter": {"avg_temp_f": 18, "rain_chance_pct": 30, "wind_mph": 22, "filming_rating": 1, "notes": "Arctic blasts, dangerous wind chill"},
    },
    # Wyoming — mountain, plains, desert
    ("WY", "mountain"): {
        "spring": {"avg_temp_f": 38, "rain_chance_pct": 35, "wind_mph": 18, "filming_rating": 3, "notes": "Snow lingers, wind strong"},
        "summer": {"avg_temp_f": 65, "rain_chance_pct": 25, "wind_mph": 12, "filming_rating": 5, "notes": "Cool and clear at elevation"},
        "fall":   {"avg_temp_f": 40, "rain_chance_pct": 15, "wind_mph": 15, "filming_rating": 4, "notes": "Aspens turning, crisp air"},
        "winter": {"avg_temp_f": 15, "rain_chance_pct": 40, "wind_mph": 25, "filming_rating": 1, "notes": "Extreme cold and wind"},
    },
    ("WY", "plains"): {
        "spring": {"avg_temp_f": 45, "rain_chance_pct": 30, "wind_mph": 22, "filming_rating": 2, "notes": "Very windy, challenging for audio"},
        "summer": {"avg_temp_f": 80, "rain_chance_pct": 20, "wind_mph": 15, "filming_rating": 4, "notes": "Hot days, thunderstorms possible"},
        "fall":   {"avg_temp_f": 48, "rain_chance_pct": 15, "wind_mph": 18, "filming_rating": 4, "notes": "Less wind, golden grasses"},
        "winter": {"avg_temp_f": 20, "rain_chance_pct": 25, "wind_mph": 25, "filming_rating": 1, "notes": "Brutal wind chill, ground blizzards"},
    },
    ("WY", "desert"): {
        "spring": {"avg_temp_f": 48, "rain_chance_pct": 20, "wind_mph": 18, "filming_rating": 3, "notes": "Dust storms possible"},
        "summer": {"avg_temp_f": 85, "rain_chance_pct": 12, "wind_mph": 12, "filming_rating": 4, "notes": "Hot, dramatic landscape"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 10, "wind_mph": 15, "filming_rating": 5, "notes": "Ideal temps, warm light"},
        "winter": {"avg_temp_f": 22, "rain_chance_pct": 20, "wind_mph": 20, "filming_rating": 1, "notes": "Cold, exposed terrain"},
    },
    # Nebraska — plains, farmland, river_valley
    ("NE", "plains"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 35, "wind_mph": 18, "filming_rating": 3, "notes": "Tornado season starts, dramatic skies"},
        "summer": {"avg_temp_f": 82, "rain_chance_pct": 30, "wind_mph": 12, "filming_rating": 4, "notes": "Hot, thunderstorm backdrops"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 14, "filming_rating": 5, "notes": "Harvest, golden light, less wind"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 20, "wind_mph": 18, "filming_rating": 2, "notes": "Cold, ice possible"},
    },
    ("NE", "farmland"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 35, "wind_mph": 15, "filming_rating": 3, "notes": "Planting season, green emerging"},
        "summer": {"avg_temp_f": 82, "rain_chance_pct": 30, "wind_mph": 10, "filming_rating": 4, "notes": "Corn fields, hot"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 18, "wind_mph": 12, "filming_rating": 5, "notes": "Harvest, combines, golden hour"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 18, "wind_mph": 15, "filming_rating": 2, "notes": "Bare fields, stark beauty"},
    },
    ("NE", "river_valley"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 38, "wind_mph": 12, "filming_rating": 4, "notes": "Platte River migration season"},
        "summer": {"avg_temp_f": 80, "rain_chance_pct": 28, "wind_mph": 10, "filming_rating": 4, "notes": "Lush, green, humid"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 18, "wind_mph": 10, "filming_rating": 5, "notes": "Reflections, fall color"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 20, "wind_mph": 14, "filming_rating": 2, "notes": "Ice fog, frozen waterways"},
    },
    # Iowa — farmland, river_valley
    ("IA", "farmland"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 40, "wind_mph": 14, "filming_rating": 3, "notes": "Muddy roads, green rolling hills"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 35, "wind_mph": 10, "filming_rating": 4, "notes": "Lush corn, humid, thunderstorms"},
        "fall":   {"avg_temp_f": 52, "rain_chance_pct": 22, "wind_mph": 12, "filming_rating": 5, "notes": "Harvest scenes, beautiful light"},
        "winter": {"avg_temp_f": 22, "rain_chance_pct": 25, "wind_mph": 16, "filming_rating": 1, "notes": "Ice storms, bitter cold"},
    },
    ("IA", "river_valley"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 42, "wind_mph": 12, "filming_rating": 3, "notes": "Flood risk, mist on rivers"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 30, "wind_mph": 8, "filming_rating": 4, "notes": "Mississippi views, humid"},
        "fall":   {"avg_temp_f": 52, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "River bluffs, fall color"},
        "winter": {"avg_temp_f": 22, "rain_chance_pct": 22, "wind_mph": 14, "filming_rating": 1, "notes": "Frozen rivers, dramatic but cold"},
    },
    # Illinois — farmland, urban, suburban
    ("IL", "farmland"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 38, "wind_mph": 14, "filming_rating": 3, "notes": "Flat, green, storm season"},
        "summer": {"avg_temp_f": 80, "rain_chance_pct": 30, "wind_mph": 10, "filming_rating": 4, "notes": "Hot, humid, corn country"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 12, "filming_rating": 5, "notes": "Harvest, warm light"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 25, "wind_mph": 16, "filming_rating": 2, "notes": "Cold, gray days common"},
    },
    ("IL", "urban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 35, "wind_mph": 15, "filming_rating": 3, "notes": "Windy City lives up to name"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 28, "wind_mph": 12, "filming_rating": 4, "notes": "Warm, good for street shots"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 14, "filming_rating": 4, "notes": "Crisp, colorful, good light"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 28, "wind_mph": 18, "filming_rating": 2, "notes": "Cold wind off lake, snow"},
    },
    ("IL", "suburban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 3, "notes": "Blooming, pleasant"},
        "summer": {"avg_temp_f": 80, "rain_chance_pct": 28, "wind_mph": 8, "filming_rating": 4, "notes": "Warm, good general conditions"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "Tree-lined streets, fall color"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 25, "wind_mph": 14, "filming_rating": 2, "notes": "Snow, slippery roads"},
    },
    # Indiana — farmland, suburban, urban
    ("IN", "farmland"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 40, "wind_mph": 12, "filming_rating": 3, "notes": "Flat, greening fields"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 32, "wind_mph": 8, "filming_rating": 4, "notes": "Lush, humid, thunderstorms"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 5, "notes": "Harvest, golden light"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 28, "wind_mph": 14, "filming_rating": 2, "notes": "Gray, cold, ice risk"},
    },
    ("IN", "suburban"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 38, "wind_mph": 10, "filming_rating": 3, "notes": "Blooming neighborhoods"},
        "summer": {"avg_temp_f": 80, "rain_chance_pct": 30, "wind_mph": 8, "filming_rating": 4, "notes": "Warm, good conditions"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "Colorful, pleasant"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 28, "wind_mph": 12, "filming_rating": 2, "notes": "Cold, snowy"},
    },
    ("IN", "urban"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 38, "wind_mph": 10, "filming_rating": 3, "notes": "Indianapolis spring"},
        "summer": {"avg_temp_f": 80, "rain_chance_pct": 30, "wind_mph": 8, "filming_rating": 4, "notes": "Hot, good for city shots"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 4, "notes": "Cool, clear days"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 28, "wind_mph": 12, "filming_rating": 2, "notes": "Cold, possible ice storms"},
    },
    # Ohio — farmland, suburban, urban, forest
    ("OH", "farmland"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 40, "wind_mph": 12, "filming_rating": 3, "notes": "Rolling hills, green"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 32, "wind_mph": 8, "filming_rating": 4, "notes": "Humid, lush, thunderstorms"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 5, "notes": "Beautiful fall colors"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 30, "wind_mph": 14, "filming_rating": 2, "notes": "Lake effect possible"},
    },
    ("OH", "urban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 38, "wind_mph": 12, "filming_rating": 3, "notes": "Cleveland, Toledo — lake breeze"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 28, "wind_mph": 10, "filming_rating": 4, "notes": "Warm, good for architecture"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 12, "filming_rating": 4, "notes": "Cool, clear days"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 35, "wind_mph": 16, "filming_rating": 1, "notes": "Lake effect snow, cold"},
    },
    ("OH", "forest"): {
        "spring": {"avg_temp_f": 50, "rain_chance_pct": 40, "wind_mph": 8, "filming_rating": 3, "notes": "Wildflowers, muddy trails"},
        "summer": {"avg_temp_f": 75, "rain_chance_pct": 28, "wind_mph": 6, "filming_rating": 4, "notes": "Dense canopy, good shade"},
        "fall":   {"avg_temp_f": 52, "rain_chance_pct": 20, "wind_mph": 8, "filming_rating": 5, "notes": "Peak fall colors"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 30, "wind_mph": 10, "filming_rating": 2, "notes": "Bare trees, frost"},
    },
    ("OH", "suburban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 38, "wind_mph": 10, "filming_rating": 3, "notes": "Blooming suburbs"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 28, "wind_mph": 8, "filming_rating": 4, "notes": "Warm, good conditions"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "Tree-lined streets, color"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 30, "wind_mph": 12, "filming_rating": 2, "notes": "Snow, ice risk"},
    },
    # Pennsylvania — mountain, forest, farmland, urban
    ("PA", "mountain"): {
        "spring": {"avg_temp_f": 48, "rain_chance_pct": 40, "wind_mph": 12, "filming_rating": 3, "notes": "Appalachian spring, rhododendron"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 30, "wind_mph": 8, "filming_rating": 4, "notes": "Green ridges, thunderstorms"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 5, "notes": "Spectacular fall colors"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 35, "wind_mph": 14, "filming_rating": 2, "notes": "Snow on ridges, fog in valleys"},
    },
    ("PA", "forest"): {
        "spring": {"avg_temp_f": 50, "rain_chance_pct": 40, "wind_mph": 8, "filming_rating": 3, "notes": "Emerging canopy, streams high"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 28, "wind_mph": 6, "filming_rating": 4, "notes": "Dense, humid, fireflies"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 20, "wind_mph": 8, "filming_rating": 5, "notes": "Peak PA fall foliage"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 30, "wind_mph": 10, "filming_rating": 2, "notes": "Bare hardwoods, frost"},
    },
    ("PA", "farmland"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 38, "wind_mph": 10, "filming_rating": 3, "notes": "Rolling green, Amish country"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 30, "wind_mph": 8, "filming_rating": 4, "notes": "Lush, pastoral scenes"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "Harvest, pumpkin patches"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 28, "wind_mph": 12, "filming_rating": 2, "notes": "Quiet, snowy fields"},
    },
    ("PA", "urban"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 35, "wind_mph": 10, "filming_rating": 3, "notes": "City blooming, pleasant"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 28, "wind_mph": 8, "filming_rating": 4, "notes": "Warm, good for city shots"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 4, "notes": "Cool, clear, crisp"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 28, "wind_mph": 12, "filming_rating": 2, "notes": "Cold, possible nor'easter snow"},
    },
    # New York — mountain, forest, urban, suburban, lake
    ("NY", "mountain"): {
        "spring": {"avg_temp_f": 45, "rain_chance_pct": 40, "wind_mph": 12, "filming_rating": 3, "notes": "Catskills/Adirondacks thaw"},
        "summer": {"avg_temp_f": 70, "rain_chance_pct": 28, "wind_mph": 8, "filming_rating": 5, "notes": "Cool mountains, clear"},
        "fall":   {"avg_temp_f": 48, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 5, "notes": "Famous fall foliage"},
        "winter": {"avg_temp_f": 22, "rain_chance_pct": 40, "wind_mph": 16, "filming_rating": 1, "notes": "Heavy snow, cold"},
    },
    ("NY", "forest"): {
        "spring": {"avg_temp_f": 48, "rain_chance_pct": 38, "wind_mph": 8, "filming_rating": 3, "notes": "Streams, wildflowers"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 25, "wind_mph": 6, "filming_rating": 4, "notes": "Lush canopy, pleasant"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 20, "wind_mph": 8, "filming_rating": 5, "notes": "NY fall at its finest"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 2, "notes": "Snow-covered, quiet"},
    },
    ("NY", "urban"): {
        "spring": {"avg_temp_f": 55, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 3, "notes": "Albany, Syracuse — city filming"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 28, "wind_mph": 8, "filming_rating": 4, "notes": "Warm, good conditions"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 4, "notes": "Cool, urban fall color"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 35, "wind_mph": 14, "filming_rating": 1, "notes": "Lake effect, cold, slippery"},
    },
    ("NY", "lake"): {
        "spring": {"avg_temp_f": 45, "rain_chance_pct": 38, "wind_mph": 14, "filming_rating": 3, "notes": "Finger Lakes, misty mornings"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 5, "notes": "Gorgeous lake reflections"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 20, "wind_mph": 12, "filming_rating": 5, "notes": "Wine country fall colors"},
        "winter": {"avg_temp_f": 25, "rain_chance_pct": 40, "wind_mph": 16, "filming_rating": 1, "notes": "Lake effect snow bands"},
    },
    ("NY", "suburban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 35, "wind_mph": 10, "filming_rating": 3, "notes": "Suburban bloom"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 25, "wind_mph": 8, "filming_rating": 4, "notes": "Warm, pleasant"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "Beautiful leaf-peeping"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 32, "wind_mph": 12, "filming_rating": 2, "notes": "Snow, quiet streets"},
    },
    # Massachusetts — coastal, urban, suburban, forest
    ("MA", "coastal"): {
        "spring": {"avg_temp_f": 50, "rain_chance_pct": 38, "wind_mph": 15, "filming_rating": 3, "notes": "Atlantic coast, fog, lobster boats"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 20, "wind_mph": 12, "filming_rating": 5, "notes": "Cape Cod vibes, warm, clear"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 25, "wind_mph": 14, "filming_rating": 4, "notes": "Dramatic coast, fall light"},
        "winter": {"avg_temp_f": 32, "rain_chance_pct": 35, "wind_mph": 18, "filming_rating": 2, "notes": "Nor'easters, cold wind"},
    },
    ("MA", "urban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 3, "notes": "Boston blooming, Freedom Trail"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 22, "wind_mph": 10, "filming_rating": 4, "notes": "Warm, tourist season"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 22, "wind_mph": 12, "filming_rating": 5, "notes": "Boston fall, historic beauty"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 30, "wind_mph": 14, "filming_rating": 2, "notes": "Cold, snow, icy sidewalks"},
    },
    ("MA", "suburban"): {
        "spring": {"avg_temp_f": 52, "rain_chance_pct": 35, "wind_mph": 10, "filming_rating": 3, "notes": "New England charm emerging"},
        "summer": {"avg_temp_f": 78, "rain_chance_pct": 22, "wind_mph": 8, "filming_rating": 4, "notes": "Pleasant, green"},
        "fall":   {"avg_temp_f": 55, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 5, "notes": "New England fall colors"},
        "winter": {"avg_temp_f": 30, "rain_chance_pct": 28, "wind_mph": 12, "filming_rating": 2, "notes": "Snowy, charming but cold"},
    },
    ("MA", "forest"): {
        "spring": {"avg_temp_f": 48, "rain_chance_pct": 38, "wind_mph": 8, "filming_rating": 3, "notes": "Berkshires awakening"},
        "summer": {"avg_temp_f": 72, "rain_chance_pct": 25, "wind_mph": 6, "filming_rating": 4, "notes": "Lush, cool shade"},
        "fall":   {"avg_temp_f": 50, "rain_chance_pct": 20, "wind_mph": 8, "filming_rating": 5, "notes": "Peak New England foliage"},
        "winter": {"avg_temp_f": 28, "rain_chance_pct": 30, "wind_mph": 10, "filming_rating": 2, "notes": "Quiet, snowy woods"},
    },
}

# Fallback for state/terrain combos not explicitly listed
_DEFAULT_WEATHER = {
    "spring": {"avg_temp_f": 50, "rain_chance_pct": 35, "wind_mph": 12, "filming_rating": 3, "notes": "Variable spring conditions"},
    "summer": {"avg_temp_f": 78, "rain_chance_pct": 25, "wind_mph": 10, "filming_rating": 4, "notes": "Generally good filming weather"},
    "fall":   {"avg_temp_f": 52, "rain_chance_pct": 20, "wind_mph": 10, "filming_rating": 4, "notes": "Cool, pleasant conditions"},
    "winter": {"avg_temp_f": 28, "rain_chance_pct": 30, "wind_mph": 15, "filming_rating": 2, "notes": "Cold, variable conditions"},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A section of US Route 20 to film.

    Each segment covers a stretch of road defined by mile markers,
    with location info, filming status, and planning metadata.
    """
    segment_id: str
    mile_marker_start: float
    mile_marker_end: float
    location_name: str
    state: str
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    description: str = ""
    terrain_type: str = "plains"
    points_of_interest: list[str] = field(default_factory=list)
    filmed: bool = False
    filming_notes: str = ""
    weather_conditions: str = ""
    best_season: str = "summer"
    priority: int = 2  # 1=must-have, 2=important, 3=nice-to-have
    planned_date: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def miles(self) -> float:
        """Total miles in this segment."""
        return abs(self.mile_marker_end - self.mile_marker_start)

    @property
    def short_id(self) -> str:
        return self.segment_id[:8]

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "mile_marker_start": self.mile_marker_start,
            "mile_marker_end": self.mile_marker_end,
            "location_name": self.location_name,
            "state": self.state,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "description": self.description,
            "terrain_type": self.terrain_type,
            "points_of_interest": self.points_of_interest,
            "filmed": self.filmed,
            "filming_notes": self.filming_notes,
            "weather_conditions": self.weather_conditions,
            "best_season": self.best_season,
            "priority": self.priority,
            "planned_date": self.planned_date,
            "miles": self.miles,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Segment":
        try:
            poi = json.loads(row["points_of_interest"]) if row["points_of_interest"] else []
        except (json.JSONDecodeError, TypeError):
            poi = []
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return cls(
            segment_id=row["segment_id"],
            mile_marker_start=row["mile_marker_start"],
            mile_marker_end=row["mile_marker_end"],
            location_name=row["location_name"],
            state=row["state"],
            gps_lat=row["gps_lat"] or 0.0,
            gps_lon=row["gps_lon"] or 0.0,
            description=row["description"] or "",
            terrain_type=row["terrain_type"] or "plains",
            points_of_interest=poi,
            filmed=bool(row["filmed"]),
            filming_notes=row["filming_notes"] or "",
            weather_conditions=row["weather_conditions"] or "",
            best_season=row["best_season"] or "summer",
            priority=row["priority"] or 2,
            planned_date=row["planned_date"] or "",
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
            metadata=meta,
        )


@dataclass
class Shot:
    """A single shot in the shot list for a segment.

    Tracks what to film, equipment needed, timing, and completion status.
    """
    shot_id: str
    segment_id: str
    shot_type: str = ""       # e.g. "wide", "tracking", "detail", "drone", "timelapse"
    description: str = ""
    equipment: str = ""       # e.g. "GoPro", "Drone", "Main camera"
    time_of_day: str = ""     # e.g. "golden_hour", "midday", "dawn", "dusk"
    duration_sec: int = 0
    status: str = "planned"   # planned, captured, reviewed, final
    notes: str = ""
    ai_generated: bool = False
    created_at: str = ""

    @property
    def short_id(self) -> str:
        return self.shot_id[:8]

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "segment_id": self.segment_id,
            "shot_type": self.shot_type,
            "description": self.description,
            "equipment": self.equipment,
            "time_of_day": self.time_of_day,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "notes": self.notes,
            "ai_generated": self.ai_generated,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Shot":
        return cls(
            shot_id=row["shot_id"],
            segment_id=row["segment_id"],
            shot_type=row["shot_type"] or "",
            description=row["description"] or "",
            equipment=row["equipment"] or "",
            time_of_day=row["time_of_day"] or "",
            duration_sec=row["duration_sec"] or 0,
            status=row["status"] or "planned",
            notes=row["notes"] or "",
            ai_generated=bool(row["ai_generated"]),
            created_at=row["created_at"] or "",
        )


# ---------------------------------------------------------------------------
# Highway20Planner
# ---------------------------------------------------------------------------

class Highway20Planner:
    """Production planner for the US Route 20 documentary.

    Manages route segments, shot lists, weather planning, and scouting notes.
    SQLite-backed with async on_change callback for dashboard integration.

    Args:
        db_path:    Path to the SQLite database file.
        router:     Optional ModelRouter for AI shot list generation.
        ltm:        Optional LongTermMemory for scouting notes.
        on_change:  Optional async callback fired after any state mutation.
    """

    def __init__(
        self,
        db_path: str = "data/highway20.db",
        router: Any = None,
        ltm: Any = None,
        on_change: Callable[..., Coroutine] | None = None,
    ):
        self.db_path = db_path
        self.router = router
        self.ltm = ltm
        self.on_change = on_change

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("Highway20Planner initialized (db=%s)", db_path)

    def _create_tables(self):
        """Create segments and shot_list tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS segments (
                segment_id       TEXT PRIMARY KEY,
                mile_marker_start REAL NOT NULL,
                mile_marker_end  REAL NOT NULL,
                location_name    TEXT NOT NULL,
                state            TEXT NOT NULL,
                gps_lat          REAL DEFAULT 0.0,
                gps_lon          REAL DEFAULT 0.0,
                description      TEXT DEFAULT '',
                terrain_type     TEXT DEFAULT 'plains',
                points_of_interest TEXT DEFAULT '[]',
                filmed           INTEGER DEFAULT 0,
                filming_notes    TEXT DEFAULT '',
                weather_conditions TEXT DEFAULT '',
                best_season      TEXT DEFAULT 'summer',
                priority         INTEGER DEFAULT 2,
                planned_date     TEXT DEFAULT '',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                metadata         TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_segments_state ON segments(state);
            CREATE INDEX IF NOT EXISTS idx_segments_filmed ON segments(filmed);
            CREATE INDEX IF NOT EXISTS idx_segments_priority ON segments(priority);

            CREATE TABLE IF NOT EXISTS shot_list (
                shot_id      TEXT PRIMARY KEY,
                segment_id   TEXT NOT NULL REFERENCES segments(segment_id),
                shot_type    TEXT DEFAULT '',
                description  TEXT DEFAULT '',
                equipment    TEXT DEFAULT '',
                time_of_day  TEXT DEFAULT '',
                duration_sec INTEGER DEFAULT 0,
                status       TEXT DEFAULT 'planned',
                notes        TEXT DEFAULT '',
                ai_generated INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_shots_segment ON shot_list(segment_id);
            CREATE INDEX IF NOT EXISTS idx_shots_status ON shot_list(status);
        """)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _notify(self):
        """Fire the on_change callback if set."""
        if self.on_change:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.on_change())
                else:
                    loop.run_until_complete(self.on_change())
            except RuntimeError:
                pass

    # ----- Segment CRUD -----

    def add_segment(
        self,
        location_name: str,
        state: str,
        mile_marker_start: float,
        mile_marker_end: float,
        **kwargs,
    ) -> Segment:
        """Add a new route segment.

        Required: location_name, state, mile_marker_start, mile_marker_end.
        Optional kwargs: gps_lat, gps_lon, description, terrain_type,
            points_of_interest, best_season, priority, planned_date, metadata.

        Returns the created Segment.
        """
        now = self._now()
        segment_id = str(uuid.uuid4())

        poi = kwargs.get("points_of_interest", [])
        if isinstance(poi, str):
            try:
                poi = json.loads(poi)
            except json.JSONDecodeError:
                poi = [p.strip() for p in poi.split(",") if p.strip()]

        meta = kwargs.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}

        self._conn.execute(
            """INSERT INTO segments
               (segment_id, mile_marker_start, mile_marker_end, location_name,
                state, gps_lat, gps_lon, description, terrain_type,
                points_of_interest, filmed, filming_notes, weather_conditions,
                best_season, priority, planned_date, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', '', ?, ?, ?, ?, ?, ?)""",
            (
                segment_id, mile_marker_start, mile_marker_end, location_name,
                state,
                kwargs.get("gps_lat", 0.0),
                kwargs.get("gps_lon", 0.0),
                kwargs.get("description", ""),
                kwargs.get("terrain_type", "plains"),
                json.dumps(poi),
                kwargs.get("best_season", "summer"),
                kwargs.get("priority", 2),
                kwargs.get("planned_date", ""),
                now, now,
                json.dumps(meta),
            ),
        )
        self._conn.commit()

        seg = self.get_segment(segment_id)
        logger.info("Segment added: %s (%s, %s) — %.1f mi",
                     location_name, state, segment_id[:8], seg.miles)
        self._notify()
        return seg

    def update_segment(self, segment_id: str, **fields) -> Segment | None:
        """Update segment fields. Returns updated Segment or None if not found."""
        allowed = {
            "mile_marker_start", "mile_marker_end", "location_name", "state",
            "gps_lat", "gps_lon", "description", "terrain_type",
            "points_of_interest", "filming_notes", "weather_conditions",
            "best_season", "priority", "planned_date", "metadata",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_segment(segment_id)

        # Serialize JSON fields
        if "points_of_interest" in updates:
            v = updates["points_of_interest"]
            if isinstance(v, list):
                updates["points_of_interest"] = json.dumps(v)
        if "metadata" in updates:
            v = updates["metadata"]
            if isinstance(v, dict):
                updates["metadata"] = json.dumps(v)

        now = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [now, segment_id]

        cursor = self._conn.execute(
            f"UPDATE segments SET {set_clause}, updated_at = ? WHERE segment_id = ?",
            values,
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None

        seg = self.get_segment(segment_id)
        logger.info("Segment updated: %s (%s)", seg.location_name, seg.short_id)
        self._notify()
        return seg

    def delete_segment(self, segment_id: str) -> bool:
        """Delete a segment and its shots. Returns True if deleted."""
        self._conn.execute("DELETE FROM shot_list WHERE segment_id = ?", (segment_id,))
        cursor = self._conn.execute(
            "DELETE FROM segments WHERE segment_id = ?", (segment_id,)
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False
        logger.info("Segment deleted: %s", segment_id[:8])
        self._notify()
        return True

    def get_segment(self, segment_id: str) -> Segment | None:
        """Fetch a single segment by ID."""
        row = self._conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        if not row:
            return None
        return Segment.from_row(row)

    def list_segments(self, state: str = "", filmed: bool | None = None) -> list[Segment]:
        """List segments with optional state and filmed filters.

        Args:
            state:  Filter by state code (e.g. "OR"). Empty = all states.
            filmed: True = only filmed, False = only unfilmed, None = all.
        """
        query = "SELECT * FROM segments WHERE 1=1"
        params: list = []
        if state:
            query += " AND state = ?"
            params.append(state)
        if filmed is not None:
            query += " AND filmed = ?"
            params.append(1 if filmed else 0)
        query += " ORDER BY mile_marker_start"
        rows = self._conn.execute(query, params).fetchall()
        return [Segment.from_row(r) for r in rows]

    def mark_filmed(
        self,
        segment_id: str,
        filming_notes: str = "",
        weather_conditions: str = "",
    ) -> Segment | None:
        """Mark a segment as filmed with optional notes.

        Returns the updated Segment or None if not found.
        """
        now = self._now()
        cursor = self._conn.execute(
            """UPDATE segments
               SET filmed = 1, filming_notes = ?, weather_conditions = ?, updated_at = ?
               WHERE segment_id = ?""",
            (filming_notes, weather_conditions, now, segment_id),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None

        seg = self.get_segment(segment_id)
        logger.info("Segment marked filmed: %s (%s)", seg.location_name, seg.short_id)
        self._notify()
        return seg

    # ----- Planning & Progress -----

    def plan_segments(self) -> dict:
        """Analyze all segments and return planning summary.

        Returns dict with total_miles, gap analysis, by-state and by-priority
        breakdowns, and schedule info.
        """
        segments = self.list_segments()
        if not segments:
            return {
                "total_segments": 0, "total_miles": 0,
                "coverage_gaps": [], "by_state": {}, "by_priority": {},
                "schedule": [],
            }

        total_miles = sum(s.miles for s in segments)

        # By-state breakdown
        by_state: dict[str, dict] = {}
        for s in segments:
            if s.state not in by_state:
                by_state[s.state] = {"segments": 0, "miles": 0, "filmed": 0}
            by_state[s.state]["segments"] += 1
            by_state[s.state]["miles"] += s.miles
            if s.filmed:
                by_state[s.state]["filmed"] += 1

        # By-priority breakdown
        by_priority: dict[int, dict] = {}
        for s in segments:
            if s.priority not in by_priority:
                by_priority[s.priority] = {"segments": 0, "miles": 0, "filmed": 0}
            by_priority[s.priority]["segments"] += 1
            by_priority[s.priority]["miles"] += s.miles
            if s.filmed:
                by_priority[s.priority]["filmed"] += 1

        # Coverage gaps — states with no segments
        covered_states = set(s.state for s in segments)
        gaps = [st for st in HWY20_STATES if st not in covered_states]

        # Schedule — segments with planned dates, sorted by date
        scheduled = sorted(
            [s.to_dict() for s in segments if s.planned_date],
            key=lambda x: x["planned_date"],
        )

        return {
            "total_segments": len(segments),
            "total_miles": round(total_miles, 1),
            "coverage_gaps": gaps,
            "by_state": by_state,
            "by_priority": {str(k): v for k, v in sorted(by_priority.items())},
            "schedule": scheduled,
        }

    def track_progress(self) -> dict:
        """Track overall filming progress.

        Returns dict with percent_complete, miles filmed/total,
        by-state and by-terrain breakdowns, upcoming dates, and estimate.
        """
        segments = self.list_segments()
        if not segments:
            return {
                "percent_complete": 0, "segments_filmed": 0, "segments_total": 0,
                "miles_filmed": 0, "miles_total": 0,
                "by_state": {}, "by_terrain": {},
                "upcoming_planned": [], "estimated_days_remaining": 0,
            }

        filmed = [s for s in segments if s.filmed]
        total_miles = sum(s.miles for s in segments)
        filmed_miles = sum(s.miles for s in filmed)

        # By-state progress
        by_state: dict[str, dict] = {}
        for st in HWY20_STATES:
            state_segs = [s for s in segments if s.state == st]
            if state_segs:
                state_filmed = sum(1 for s in state_segs if s.filmed)
                by_state[st] = {
                    "total": len(state_segs),
                    "filmed": state_filmed,
                    "miles": round(sum(s.miles for s in state_segs), 1),
                    "status": "complete" if state_filmed == len(state_segs) else
                              "partial" if state_filmed > 0 else "empty",
                }
            else:
                by_state[st] = {"total": 0, "filmed": 0, "miles": 0, "status": "empty"}

        # By-terrain progress
        by_terrain: dict[str, dict] = {}
        for s in segments:
            t = s.terrain_type
            if t not in by_terrain:
                by_terrain[t] = {"total": 0, "filmed": 0, "miles": 0}
            by_terrain[t]["total"] += 1
            by_terrain[t]["miles"] += s.miles
            if s.filmed:
                by_terrain[t]["filmed"] += 1

        # Upcoming planned dates
        upcoming = sorted(
            [{"segment_id": s.segment_id, "location": s.location_name,
              "state": s.state, "date": s.planned_date}
             for s in segments if s.planned_date and not s.filmed],
            key=lambda x: x["date"],
        )[:10]

        # Rough estimate: average 2 segments per filming day
        remaining = len(segments) - len(filmed)
        est_days = max(1, remaining // 2) if remaining > 0 else 0

        pct = round((len(filmed) / len(segments)) * 100, 1) if segments else 0

        return {
            "percent_complete": pct,
            "segments_filmed": len(filmed),
            "segments_total": len(segments),
            "miles_filmed": round(filmed_miles, 1),
            "miles_total": round(total_miles, 1),
            "by_state": by_state,
            "by_terrain": by_terrain,
            "upcoming_planned": upcoming,
            "estimated_days_remaining": est_days,
        }

    # ----- AI Shot List Generation -----

    async def shot_list_generator(self, segment_id: str) -> list[Shot]:
        """Generate an AI shot list for a segment using the local model.

        Builds a prompt from segment data, calls the router with
        task_complexity="routine" (local model, free), parses the JSON
        response, and falls back to terrain-aware defaults if parsing fails.

        Returns a list of Shot objects (already saved to DB).
        """
        seg = self.get_segment(segment_id)
        if not seg:
            return []

        shots = []

        if self.router:
            prompt = self._build_shot_prompt(seg)
            try:
                response = await self.router.route(prompt, task_complexity="routine")
                shots = self._parse_shot_response(response, segment_id)
            except Exception as e:
                logger.warning("AI shot generation failed for %s: %s — using defaults",
                             seg.short_id, e)

        # Fallback to terrain-aware defaults if AI didn't produce shots
        if not shots:
            shots = self._terrain_default_shots(seg)

        # Save all shots to DB
        now = self._now()
        for shot in shots:
            shot.created_at = now
            self._conn.execute(
                """INSERT INTO shot_list
                   (shot_id, segment_id, shot_type, description, equipment,
                    time_of_day, duration_sec, status, notes, ai_generated, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?)""",
                (
                    shot.shot_id, shot.segment_id, shot.shot_type,
                    shot.description, shot.equipment, shot.time_of_day,
                    shot.duration_sec, shot.notes,
                    1 if shot.ai_generated else 0, now,
                ),
            )
        self._conn.commit()

        logger.info("Generated %d shots for segment %s (%s)",
                     len(shots), seg.location_name, seg.short_id)
        self._notify()
        return shots

    def _build_shot_prompt(self, seg: Segment) -> str:
        """Build the prompt for AI shot list generation."""
        poi_str = ", ".join(seg.points_of_interest) if seg.points_of_interest else "none noted"
        return f"""Generate a shot list for a motorcycle documentary segment.

Location: {seg.location_name}, {seg.state}
Terrain: {seg.terrain_type}
Mile markers: {seg.mile_marker_start} to {seg.mile_marker_end} ({seg.miles:.1f} miles)
Description: {seg.description or 'No description'}
Points of interest: {poi_str}
Best season: {seg.best_season}

Return a JSON array of 4-6 shots. Each shot object should have:
- "shot_type": one of "wide", "tracking", "detail", "drone", "timelapse", "pov", "interview_setup"
- "description": what to capture (1-2 sentences)
- "equipment": camera/gear needed
- "time_of_day": "dawn", "golden_hour", "midday", "afternoon", "dusk", "any"
- "duration_sec": estimated duration in seconds (5-120)
- "notes": any special considerations

Return ONLY the JSON array, no other text."""

    def _parse_shot_response(self, response: str, segment_id: str) -> list[Shot]:
        """Parse AI response into Shot objects."""
        # Try to find JSON array in the response
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []

        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []

        shots = []
        for item in data:
            if not isinstance(item, dict):
                continue
            shot = Shot(
                shot_id=str(uuid.uuid4()),
                segment_id=segment_id,
                shot_type=item.get("shot_type", "wide"),
                description=item.get("description", ""),
                equipment=item.get("equipment", ""),
                time_of_day=item.get("time_of_day", "any"),
                duration_sec=int(item.get("duration_sec", 30)),
                notes=item.get("notes", ""),
                ai_generated=True,
            )
            shots.append(shot)
        return shots

    def _terrain_default_shots(self, seg: Segment) -> list[Shot]:
        """Generate terrain-aware default shots when AI is unavailable."""
        defaults = [
            Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                 shot_type="wide", description=f"Wide establishing shot of {seg.location_name}",
                 equipment="Main camera + wide lens", time_of_day="golden_hour",
                 duration_sec=15, ai_generated=False),
            Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                 shot_type="tracking", description="Riding shot along the road",
                 equipment="GoPro mounted", time_of_day="any",
                 duration_sec=60, ai_generated=False),
            Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                 shot_type="detail", description=f"Road sign and mile marker detail",
                 equipment="Main camera", time_of_day="any",
                 duration_sec=10, ai_generated=False),
        ]

        # Terrain-specific shots
        terrain_shots = {
            "mountain": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                           shot_type="drone", description="Aerial of mountain road switchbacks",
                           equipment="Drone", time_of_day="morning",
                           duration_sec=30, ai_generated=False),
            "coastal": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                          shot_type="timelapse", description="Ocean waves and coastal road timelapse",
                          equipment="Main camera + tripod", time_of_day="dusk",
                          duration_sec=20, ai_generated=False),
            "desert": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                         shot_type="timelapse", description="Heat shimmer and open desert road",
                         equipment="Main camera + tripod", time_of_day="midday",
                         duration_sec=20, ai_generated=False),
            "forest": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                         shot_type="pov", description="Riding through tree canopy, dappled light",
                         equipment="GoPro chest mount", time_of_day="morning",
                         duration_sec=45, ai_generated=False),
            "plains": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                         shot_type="drone", description="Aerial of straight road to horizon",
                         equipment="Drone", time_of_day="golden_hour",
                         duration_sec=30, ai_generated=False),
            "farmland": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                           shot_type="wide", description="Farmland panorama with road cutting through",
                           equipment="Main camera + wide lens", time_of_day="golden_hour",
                           duration_sec=15, ai_generated=False),
            "urban": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                        shot_type="tracking", description="City street riding, traffic and architecture",
                        equipment="GoPro mounted", time_of_day="afternoon",
                        duration_sec=45, ai_generated=False),
            "river_valley": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                               shot_type="drone", description="River valley aerial with road following water",
                               equipment="Drone", time_of_day="morning",
                               duration_sec=30, ai_generated=False),
            "lake": Shot(shot_id=str(uuid.uuid4()), segment_id=seg.segment_id,
                        shot_type="wide", description="Lake reflection shot with motorcycle parked",
                        equipment="Main camera + polarizer", time_of_day="dawn",
                        duration_sec=15, ai_generated=False),
        }

        terrain_shot = terrain_shots.get(seg.terrain_type)
        if terrain_shot:
            defaults.append(terrain_shot)

        return defaults

    # ----- Weather Planner -----

    def weather_planner(self, segment_id: str = "") -> list[dict]:
        """Get seasonal weather/filming conditions for segments.

        If segment_id is given, returns weather for that segment only.
        Otherwise returns weather for all unfilmed segments.

        Each result includes seasonal data and flags mismatches with
        the segment's best_season setting.
        """
        if segment_id:
            seg = self.get_segment(segment_id)
            segments = [seg] if seg else []
        else:
            segments = self.list_segments(filmed=False)

        results = []
        for seg in segments:
            key = (seg.state, seg.terrain_type)
            weather = SEASONAL_WEATHER.get(key, _DEFAULT_WEATHER)

            # Check for best_season mismatch
            best = seg.best_season.lower()
            best_rating = weather.get(best, {}).get("filming_rating", 3)
            # Find the actual best season by rating
            actual_best = max(weather.items(), key=lambda x: x[1].get("filming_rating", 0))

            mismatch = (best_rating < actual_best[1].get("filming_rating", 0))

            results.append({
                "segment_id": seg.segment_id,
                "location_name": seg.location_name,
                "state": seg.state,
                "terrain_type": seg.terrain_type,
                "best_season": seg.best_season,
                "seasons": weather,
                "mismatch": mismatch,
                "recommended_season": actual_best[0] if mismatch else seg.best_season,
            })

        return results

    # ----- Shot CRUD -----

    def add_shot(
        self,
        segment_id: str,
        shot_type: str = "wide",
        description: str = "",
        **kwargs,
    ) -> Shot | None:
        """Manually add a shot to a segment's shot list."""
        seg = self.get_segment(segment_id)
        if not seg:
            return None

        now = self._now()
        shot_id = str(uuid.uuid4())

        self._conn.execute(
            """INSERT INTO shot_list
               (shot_id, segment_id, shot_type, description, equipment,
                time_of_day, duration_sec, status, notes, ai_generated, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                shot_id, segment_id, shot_type, description,
                kwargs.get("equipment", ""),
                kwargs.get("time_of_day", "any"),
                kwargs.get("duration_sec", 30),
                kwargs.get("status", "planned"),
                kwargs.get("notes", ""),
                now,
            ),
        )
        self._conn.commit()

        shot = self.get_shot(shot_id)
        logger.info("Shot added: %s for segment %s", shot.short_id, seg.short_id)
        self._notify()
        return shot

    def update_shot(self, shot_id: str, **fields) -> Shot | None:
        """Update shot fields. Returns updated Shot or None."""
        allowed = {"shot_type", "description", "equipment", "time_of_day",
                   "duration_sec", "status", "notes"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_shot(shot_id)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [shot_id]

        cursor = self._conn.execute(
            f"UPDATE shot_list SET {set_clause} WHERE shot_id = ?",
            values,
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None

        shot = self.get_shot(shot_id)
        logger.info("Shot updated: %s", shot.short_id)
        self._notify()
        return shot

    def delete_shot(self, shot_id: str) -> bool:
        """Delete a shot. Returns True if deleted."""
        cursor = self._conn.execute(
            "DELETE FROM shot_list WHERE shot_id = ?", (shot_id,)
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return False
        logger.info("Shot deleted: %s", shot_id[:8])
        self._notify()
        return True

    def get_shot(self, shot_id: str) -> Shot | None:
        """Fetch a single shot by ID."""
        row = self._conn.execute(
            "SELECT * FROM shot_list WHERE shot_id = ?", (shot_id,)
        ).fetchone()
        if not row:
            return None
        return Shot.from_row(row)

    def get_shots(self, segment_id: str = "") -> list[Shot]:
        """List shots, optionally filtered by segment."""
        if segment_id:
            rows = self._conn.execute(
                "SELECT * FROM shot_list WHERE segment_id = ? ORDER BY created_at",
                (segment_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM shot_list ORDER BY created_at"
            ).fetchall()
        return [Shot.from_row(r) for r in rows]

    # ----- Scouting Notes (LTM) -----

    def store_scouting_note(self, segment_id: str, note: str) -> bool:
        """Store a scouting note in long-term memory.

        Notes are tagged with source=highway20_scouting and the segment_id
        for later retrieval.
        """
        if not self.ltm:
            logger.warning("No LTM available — scouting note not stored")
            return False

        seg = self.get_segment(segment_id)
        if not seg:
            return False

        try:
            self.ltm.store(
                text=f"[Scouting: {seg.location_name}, {seg.state}] {note}",
                metadata={
                    "source": "highway20_scouting",
                    "segment_id": segment_id,
                    "state": seg.state,
                    "location": seg.location_name,
                },
            )
            logger.info("Scouting note stored for %s (%s)", seg.location_name, seg.short_id)
            return True
        except Exception as e:
            logger.error("Failed to store scouting note: %s", e)
            return False

    def search_scouting_notes(self, query: str, n_results: int = 10) -> list[dict]:
        """Search scouting notes in long-term memory.

        Returns list of dicts with text, metadata, and relevance score.
        """
        if not self.ltm:
            return []

        try:
            results = self.ltm.search(
                query=query,
                n_results=n_results,
                where={"source": "highway20_scouting"},
            )
            return results if isinstance(results, list) else []
        except Exception as e:
            logger.error("Scouting note search failed: %s", e)
            return []

    # ----- Dashboard Integration -----

    def get_status(self) -> dict:
        """Quick summary for status displays."""
        total = self._conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        filmed = self._conn.execute("SELECT COUNT(*) FROM segments WHERE filmed = 1").fetchone()[0]
        total_shots = self._conn.execute("SELECT COUNT(*) FROM shot_list").fetchone()[0]

        miles_row = self._conn.execute(
            "SELECT COALESCE(SUM(ABS(mile_marker_end - mile_marker_start)), 0) FROM segments"
        ).fetchone()
        total_miles = round(miles_row[0], 1)

        filmed_miles_row = self._conn.execute(
            "SELECT COALESCE(SUM(ABS(mile_marker_end - mile_marker_start)), 0) FROM segments WHERE filmed = 1"
        ).fetchone()
        filmed_miles = round(filmed_miles_row[0], 1)

        pct = round((filmed / total) * 100, 1) if total > 0 else 0

        return {
            "segments_total": total,
            "segments_filmed": filmed,
            "total_miles": total_miles,
            "filmed_miles": filmed_miles,
            "total_shots": total_shots,
            "percent_complete": pct,
        }

    def to_broadcast_dict(self) -> dict:
        """Full state snapshot for dashboard broadcast."""
        segments = self.list_segments()
        shots = self.get_shots()
        progress = self.track_progress()
        status = self.get_status()

        return {
            "segments": [s.to_dict() for s in segments],
            "shots": [s.to_dict() for s in shots],
            "progress": progress,
            "status": status,
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("Highway20Planner database closed")
