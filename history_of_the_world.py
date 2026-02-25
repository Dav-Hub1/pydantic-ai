import asyncio
import os
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()
model_name = os.getenv("MODEL_NAME")

agent=Agent(
    model=model_name,
    system_prompt=""" Be as verbose as possible, reply with a giant text
    and include as much information as possible.""",
)
# Be careful with this, it can generate very long responses and consume a lot of tokens, which may lead to higher costs and slower response times. Always monitor the output and adjust the prompt as needed to balance verbosity with relevance and conciseness.
async def main():
    async with agent.run_stream('Tell me about the history of the world') as result:
        async for message in result.stream_text(delta=True):
            print(message, end='', flush=True)

if __name__ == "__main__":
    asyncio.run(main())