import os
from pydantic import Field, BaseModel
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
import requests

load_dotenv()

model_name = os.getenv("TRIAGE_MODEL")

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

chat_agent=Agent(
    model=model_name,
    system_prompt="""Be as concise as possible in your responses and reply with
    less than 71 tokens.""",
    deps_type=CredentialManager,
    output_type=WeatherInfo,
)

@chat_agent.tool
def get_weather_info(ctx: RunContext[CredentialManager], latitude: float, longitude: float) -> str:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    return response.json()

message_history = []
while True:
    current_message = input("User: ")
    if current_message == 'exit':
        break
    result = chat_agent.run_sync(current_message, message_history=message_history, deps=creds_manager)
    message_history = result.new_messages()
    print(f"Agent: {result.output.model_dump_json(indent=4)}")