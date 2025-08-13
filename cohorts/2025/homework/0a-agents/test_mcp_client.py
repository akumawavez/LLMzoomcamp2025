import json
import weather_mcp_server
import mcp_client

def main():
    our_mcp_client = mcp_client.MCPClient(["python", "weather_mcp_server.py"])
    our_mcp_client.start_server()
    our_mcp_client.initialize()
    our_mcp_client.initialized()

    mcp_tools = mcp_client.MCPTools(mcp_client=our_mcp_client)

    print(mcp_tools.get_tools())

    result_berlin = our_mcp_client.call_tool('get_weather', {'city': 'Berlin'})
    print(result_berlin)

    developer_prompt = """
    You help users find out the weather in their cities. 
    If they didn't specify a city, ask them. Make sure we always use a city.
    """.strip()

    import chat_assistant

    chat_interface = chat_assistant.ChatInterface()

    chat = chat_assistant.ChatAssistant(
        tools=mcp_tools,
        developer_prompt=developer_prompt,
        chat_interface=chat_interface,
        client=our_mcp_client
    )

    chat.run()

if __name__ == "__main__":
    main()