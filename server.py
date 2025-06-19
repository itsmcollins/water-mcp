from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pydantic import BaseModel
import httpx
import os
from typing import Optional, List

load_dotenv()

FLOOD_MONITORING_BASE_URL = 'https://environment.data.gov.uk/flood-monitoring'
GEOCODING_BASE_URL = 'https://geocode.maps.co/search'


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class Radius(BaseModel):
    coordinates: Coordinates
    radius: int



class FloodArea(BaseModel):
    id: str
    county: str
    description: str
    eaAreaName: str
    label: str
    lat: float
    long: float
    quickDialNumber: str
    riverOrSea: str
    


mcp = FastMCP("water")



# Resources
# ------------------------
@mcp.resource("test://hello", name="Hello", description="Hello")
def hello():
    """Hello"""
    return "Hello"


@mcp.resource("flood://areas", name="Flood Areas", description="Get flood areas")
def get_flood_areas() -> List:
    """Get flood areas"""
    response = httpx.get(
        FLOOD_MONITORING_BASE_URL + '/id/floodAreas',
    )

    return response.json()["items"]


@mcp.resource(
    uri="flood://warnings-and-alerts",
    name="Flood Warnings and Alerts",
    description="Get flood warnings and alerts."
)
def get_flood_warnings_and_alerts():
    """Get flood warnings and alerts"""
    response = httpx.get(
        FLOOD_MONITORING_BASE_URL + '/id/floods',
    )
    return response.json()["items"]







# Tools
# ------------------------

@mcp.tool(
    name="convert_location_to_coordinates",
    description="Convert a location to latitude and longitude coordinates"
)
def convert_location_to_coordinates(location: str) -> Coordinates:
    """Convert a location to latitude and longitude coordinates"""
    
    response = httpx.get(
        GEOCODING_BASE_URL,
        params={
            "q": location,
            "api_key": os.getenv("GEOCODING_API_KEY")
        }
    )

    data = response.json()
    if data:
        return Coordinates(latitude=data[0]["lat"], longitude=data[0]["lon"])
    else:
        raise ValueError("Location not found")



@mcp.tool(
    name="quick_dial_flood_area",
    description="Quick dial a flood area"
)
def quick_dial_flood_area(flood_area_id: str) -> str:
    """Quick dial a flood area"""
    response = httpx.get(
        FLOOD_MONITORING_BASE_URL + '/id/floodAreas/' + flood_area_id,
    )

    if response.status_code != 200:
        return f"No quick dial number found for flood area {flood_area_id}"

    data  = response.json()["items"]
    if data and 'quickDialNumber' in data:
        return f"Phoning flood area {flood_area_id} on {data['quickDialNumber']}..."
    else:
        return f"No quick dial number found for flood area {flood_area_id}"





# Run
# ------------------------
mcp.run(transport="streamable-http")
