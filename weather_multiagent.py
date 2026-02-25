import os
from pydantic import Field, BaseModel
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import requests

load_dotenv()

model_name = os.getenv("TRIAGE_MODEL")

############################ Weather Agent ########################


class CredentialManager:
    def __init__(self):
        pass
    def get_credentials(self):
        return {
            'username': 'test', #os.getenv("API_USERNAME")
            'password': 'secret_password', #os.getenv("API_PASSWORD")
        }

creds_manager = CredentialManager()

class WeatherInfo(BaseModel):
    temperature: float = Field(..., description="Current temperature in Celsius")
    humidity: float = Field(..., description="Current humidity percentage")
    wind_speed: float = Field(..., description="Current wind speed in km/h")

weather_agent=Agent(
    model=model_name,
    system_prompt="""Be as concise as possible in your responses and reply with
    less than 71 tokens.""",
    deps_type=CredentialManager,
    output_type=WeatherInfo,
)

@weather_agent.tool
def get_weather_info(ctx: RunContext[CredentialManager], latitude: float, longitude: float) -> str:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    return response.json()

################################## Math Agent ########################

math_agent=Agent(
    model=model_name,
    system_prompt="""You are a mathematician and great explainer.
    Be as concise as possible in your responses and reply with less than 71 tokens.""",
)

############################ Main Agent ########################

main_agent=Agent(
    model=model_name,
    system_prompt="""You are capable of delegating tasks to other agents
    if they are more specialized for a given task.""",
)

@main_agent.tool_plain
async def call_weather_agent(location: str) -> WeatherInfo:
    result = await weather_agent.run(f"Get the current weather for location {location}.", deps=creds_manager)
    return result.output

@main_agent.tool_plain
async def call_math_agent(expression: str) -> str:
    result = await math_agent.run(f"Calculate the mathematical average of {expression}.")
    return result.output

message_history = []
while True:
    current_message = input("User: ")
    if current_message == 'exit':
        break
    result = main_agent.run_sync(current_message, message_history=message_history)
    message_history = result.new_messages()
    print(f"Agent: {result.output}")