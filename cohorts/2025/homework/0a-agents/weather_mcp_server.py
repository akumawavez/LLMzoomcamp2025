import random
from fastmcp import FastMCP


# ## Q4. Simple MCP Server
# A simple MCP server from the documentation looks like that:
# weather_server.py
from fastmcp import FastMCP
mcp = FastMCP("Demo 🚀")


known_weather_data = {
    'berlin': 20.0, "paris": 22.0
}


@mcp.tool
def get_weather(city: str) -> float:
    """
    Retrieves the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        float: The temperature associated with the city.
    """
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]

    return round(random.uniform(-5, 35), 1)


@mcp.tool
def set_weather(city: str, temp: float) -> None:
    """
    Sets the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to set the weather data.
        temp (float): The temperature to associate with the city.

    Returns:
        str: A confirmation string 'OK' indicating successful update.
    """
    city = city.strip().lower()
    known_weather_data[city] = temp
    return 'OK'

# Q5
import sys
if __name__ == "__main__":
    print("Starting Weather Demo (logs -> stderr)", file=sys.stderr, flush=True)
    # mcp.run()  # stdio transport
    mcp.run()

