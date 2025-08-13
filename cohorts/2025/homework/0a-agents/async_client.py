
## Q6
# async_client.py
import asyncio
import weather_mcp_server
from fastmcp import Client

async def main():
    async with Client(weather_mcp_server.mcp) as mcp_client:
        tools = await mcp_client.list_tools()
        print(tools)

if __name__ == "__main__":
    asyncio.run(main())