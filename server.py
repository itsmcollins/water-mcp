from mcp.server.fastmcp import FastMCP, Image
from dotenv import load_dotenv
from pydantic import BaseModel
import httpx
import os
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io


# Democratise decision power, coordinate stakeholders, propose actions, act
# 1. Location --> Flood Warnings --> Rain Forecast --> Water Quality --> Assess Risk --> Draft Recommendation
# 2. My world of water --> all measurements around me :)
# 3. Leak detection
# 4. Optimal RTO Bid --> foucsed on CAISO


# Solve this problem --> aggregate data --> train model --> simulate --> take action --> write report
# Predict nitrate concentration at station xyz next week
# --> hydrology api + weather forecast api
# --> train model and get answer
# --> pretend to send an email report

# Predict drought risk in Windsor for the next 3 months
# --> hydrology api + weather forecast api
# --> train model and get answer
# --> pretend to send an email report


load_dotenv()


GEOCODING_BASE_URL = 'https://geocode.maps.co'
NWS_BASE_URL = 'https://api.weather.gov'
HYDROLOGY_BASE_URL = 'https://environment.data.gov.uk/hydrology'
METOFFICE_GLOBAL_SPOT_BASE_URL = 'https://data.hub.api.metoffice.gov.uk/sitespecific/v0'


# Models
# ------------------------

class Coordinates(BaseModel):
    latitude: float
    longitude: float



# MCP
# ------------------------
mcp = FastMCP("water")



# Resources
# ------------------------

@mcp.resource("api://nws", name="NWS OpenAPI", description="NWS OpenAPI")
def nws():
    """NWS OpenAPI"""
    return httpx.get(NWS_BASE_URL + '/openapi.json').json()


@mcp.resource("api://hydrology/uk", name="UK Hydrology OpenAPI", description="UK Hydrology OpenAPI")
def uk_hydrology():
    """UK Hydrology OpenAPI"""
    return httpx.get(HYDROLOGY_BASE_URL + '/doc/oas.json').json()






# Tools
# ------------------------

@mcp.tool(
    name="date-today",
    description="Get today's date"
)
def get_today_date() -> str:
    """Get today's date"""
    return datetime.now().strftime("%Y-%m-%d")





@mcp.tool(
    name="execute-api-call",
    description="Execute an API call"
)
def execute_api_call(
    method: str,
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    json: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> dict:
    """Execute an API call"""
    response = httpx.request(
        method,
        url,
        headers=headers,
        params=params,
        data=data,
        json=json,
        timeout=timeout,
    )
    return response.json()





class RegressorInput(BaseModel):
    training_data: Dict[str, List[float]]
    target_column: str
    train_test_split: float = 0.8
    prediction_to_run: Dict[str, List[float]]


class RegressorOutput(BaseModel):
    accuracy: float
    prediction: List[float]
    confidence: float


@mcp.tool(
    name="train_regressor",
    description="""
    Train a linear regression model on tabular data and return predictions.

    This tool trains a scikit-learn LinearRegression model on your data and makes predictions
    for new data points. The data must be provided in a column-oriented format.

    Example payload:
    ```json
    {
      "training_data": {
        "sqft": [1000, 1200, 1500, 1800, 2000],
        "bedrooms": [2, 2, 3, 3, 4],
        "price": [200000, 240000, 300000, 360000, 400000]
      },
      "target_column": "price",
      "prediction_to_run": {
        "sqft": [1600, 2300],
        "bedrooms": [3, 4]
      }
    }
    ```

    Expected output:
    ```json
    {
      "accuracy": 0.97,
      "prediction": [320000, 460000],
      "confidence": 0.96
    }
    ```

    Parameter details:
    - training_data: Dictionary where keys are column names and values are lists of data points
    - target_column: Name of the column to predict (must be one of the keys in training_data)
    - prediction_to_run: Dictionary with the same keys as training_data (except target_column)
    - train_test_split: Optional, proportion of data used for training (default: 0.8)

    Common errors:
    - "Target column X not found in training data" → Check that target_column exactly matches a key in training_data
    - "Prediction to run must have same columns as training data" → Ensure prediction_to_run has the same column names
    """
)
def train_regressor(
    input: RegressorInput
) -> RegressorOutput:
    """Train a regressor"""

    # confirm target column is in training data
    if input.target_column not in input.training_data:
        raise ValueError(f"Target column {input.target_column} not found in training data")
    

    # confirm prediction to run has same columns as training data
    if set(input.prediction_to_run.keys()) != set(input.training_data.keys() - {input.target_column}):
        raise ValueError("Prediction to run must have same columns as training data")

    df = pd.DataFrame(input.training_data)
    y = df[input.target_column]
    X = df.drop(columns=[input.target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-input.train_test_split)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prediction = model.predict(pd.DataFrame(input.prediction_to_run))
    confidence=  model.score(X_test, y_test)
    
    return RegressorOutput(
        accuracy=float(r2_score(y_test, y_pred)),
        prediction=prediction.tolist(),
        confidence=float(confidence)
    )






class PlotInput(BaseModel):
    x_axis: List[float]
    y_axis: List[float]
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    is_scatter: Optional[bool] = True


@mcp.tool(
    name="plot_data",
    description="""
    Plot x vs y data and return the figure as a PNG image.

    This tool creates a simple scatter plot from two lists of values and returns
    the visualization as an image.

    Example payload:
    ```json
    {
      "x_axis": [1000, 1200, 1500, 1800, 2000],
      "y_axis": [200000, 240000, 300000, 360000, 400000],
      "title": "House Prices by Area",
      "x_label": "Square Feet",
      "y_label": "Price ($)"
    }
    ```

    Parameter details:
    - x_axis: List of numeric values for the x-axis
    - y_axis: List of numeric values for the y-axis (must be same length as x_axis)
    - title: Optional title for the plot
    - x_label: Optional label for the x-axis
    - y_label: Optional label for the y-axis

    Common errors:
    - "Length mismatch" → Ensure x_axis and y_axis have the same number of elements
    """
)
def plot_data(
    input: PlotInput
) -> Image:
    """Plot data and return image"""

    # confirm x and y axis have same length
    if len(input.x_axis) != len(input.y_axis):
        raise ValueError("X and Y axis must have same length")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if input.is_scatter:
        ax.scatter(input.x_axis, input.y_axis)
    else:
        ax.plot(input.x_axis, input.y_axis)
    ax.grid(True)
    if input.title:
        ax.set_title(input.title)
    if input.x_label:
        ax.set_xlabel(input.x_label)
    if input.y_label:
        ax.set_ylabel(input.y_label)


    # *** Save to PNG in-memory ***
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', dpi=80, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    # Return the *encoded* PNG bytes
    return Image(data=buf.getvalue(), format='jpeg')






@mcp.tool(
    name="convert_location_to_coordinates",
    description="Convert a location to latitude and longitude coordinates"
)
def convert_location_to_coordinates(location: str) -> Coordinates:
    """Convert a location to latitude and longitude coordinates"""
    
    response = httpx.get(
        GEOCODING_BASE_URL + '/search',
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
    name='get-hourly-weather-forecast',
    description='Get hourly weather forecast'
)
def get_hourly_weather_forecast(
    latitude: float,
    longitude: float,
) -> str:
    """Get hourly weather forecast"""
    
    response = httpx.get(
        METOFFICE_GLOBAL_SPOT_BASE_URL + '/point/hourly',
        params={
            "apikey": os.getenv("METOFFICE_GLOBAL_SPOT_API_KEY"),
            "latitude": latitude,
            "longitude": longitude,
            "excludeParameterMetadata": True
        }
    )

    response.raise_for_status()
    
    return response.json()
    



# Run
# ------------------------
mcp.run(transport="streamable-http")
