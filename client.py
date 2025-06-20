from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage
from typing import List

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")

ollama_model = OpenAIModel(
    model_name='llama3.2',
    provider=OpenAIProvider(
        base_url='http://localhost:11434/v1'
    )
)

agent = Agent(
    model=ollama_model,
    mcp_servers=[server],
    system_prompt="You are a helpful assistant that solves user problems.",
    instructions="""
    First, make a judgement on what the user query is about.
    Second, determine if you have resources that might be helpful to answer that question.
    Third, if you do, use them. If not, you MUST NOT use them.
    If you do have a resource that you could use but do not have enough information to use it,
    ask the user politely to provide the information you need.
    Lastly, respond in a charming way, whether you are including the results of the tools or not. 

    At all points: do not provide any additional commentary than is demanded. Do not guide the 
    conversation further.  
    """
)


async def main():
    print("\n")
    print("Water / Electricity Grid MCP AGI Project")
    print("-----------------------------------------")
    print("Type 'q' or 'quit' to stop at any point")
    print("\n")

    message_history: List[ModelMessage] = []
    async with agent.run_mcp_servers():
        while True:
            user_input = input(">> ")
            if user_input.lower() in ['q', 'quit']:
                break
            result = await agent.run(user_input, message_history=message_history)
            print('\n', result.output, '\n')
            message_history.extend(result.new_messages())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    