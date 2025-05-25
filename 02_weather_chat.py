from agents import (
  Agent, handoff, RunContextWrapper, Runner, AsyncOpenAI, set_default_openai_client, set_default_openai_api,
  ToolCallItem,
  ToolCallOutputItem,
  MessageOutputItem,
  HandoffOutputItem,
  ItemHelpers
)
from agents.tool import function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv
import os
import asyncio
import gradio as gr
from pydantic import BaseModel

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

OPEN_WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


external_client = AsyncOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
    default_headers={"api-key": AZURE_OPENAI_API_KEY},
    default_query={"api-version": AZURE_OPENAI_API_VERSION}
)

set_default_openai_client(external_client, use_for_tracing=False)
set_default_openai_api("chat_completions")

from axios import Axios

httpClient = Axios()



class WeatherContext(BaseModel):
   city_name: str | None = None
   lat: float | None = None
   lon: float | None = None


class GeoLocation:
  def __init__(self, name, lat, lon):
    self.name = name
    self.lat = lat
    self.lon = lon

  @classmethod
  def from_json(cls, json_data: dict):
    return cls(
      name=json_data.get('name'),
      lat=json_data.get('lat'),
      lon=json_data.get('lon')
    )
  
  def to_dict(self):
    return {
      "name": self.name,
      "lat": self.lat,
      "lon": self.lon
    }

  def __repr__(self):
    return f"GeoLocation(name='{self.name}', lat='{self.lat}', lon='{self.lon}')"
  

@function_tool
async def get_geo(
  context: RunContextWrapper[WeatherContext],
  city: str):
  """
  Get geolocation details of a city.
  Parameters:
    - city (str): The name of the city to get location info for.
  Returns:
    - {"lat": float, "lon": float}: Basic location info or coordinates.
  """
  url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={OPEN_WEATHER_API_KEY}"
  response = await httpClient.aget(url)
  geo = GeoLocation.from_json(response.data[0])
  print(geo)

  context.context.city_name = city
  context.context.lat = geo.lat
  context.context.lon = geo.lon

  return geo.to_dict()


class WeatherInfo():
  def __init__(self, main, temp_min, temp_max):
    self.main = main
    self.temp_min = temp_min
    self.temp_max = temp_max

  @classmethod
  def from_json(cls, json_data: dict):
    weather = json_data.get('weather')
    main = json_data.get('main')

    return cls(
      main=weather[0].get('main'),
      temp_min=main.get('temp_min'),
      temp_max=main.get('temp_max')
    )
  
  def to_dict(self):
    return {
      "main": self.main,
      "temp_min": self.temp_min,
      "temp_max": self.temp_max
    }
  
  def __repr__(self):
    return f"WeatherInfo(main='{self.main}', min='{self.temp_min}', max='{self.temp_max}')"

 
@function_tool
async def get_forecast(
   context: RunContextWrapper[WeatherContext],
   lat: float, lon: float):
  """
  Get weather details of a city.
  Parameters:
    - lat (float): Latitude of a city.
    - lon (float): Longitude of a city.
  Returns:
    - {"main": string, "temp_min": number, "temp_max": number}: Weather info contains main, min temperature and max temperature.
  """
  url=f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPEN_WEATHER_API_KEY}&units=metric"
  response = await httpClient.aget(url)
  info = WeatherInfo.from_json(response.data)
  print(info)
  return info.to_dict()  


weather_instruction = f"""
{RECOMMENDED_PROMPT_PREFIX} \
You find the weather info using a given lat and lon. \
You only use the given tools. \
Given the lat and lon you find the weather info using get_forecast. \
You MUST transfer back to Forecast agent once you complete the task.
"""

weather_agent = Agent[WeatherContext](
  name = "Weather Agent",
  instructions = weather_instruction,
  tools = [get_forecast]
)



geo_instruction = f"""
{RECOMMENDED_PROMPT_PREFIX} \
You extract the city name from the text in any given language. \
You only use the given tools. \
You get the result using get_geo tool using the city name. \
You will handoff the lat and lon to Weather Agent. \
If you cannot answer the question, transfer back to Forecast Agent
"""

geo_agent = Agent[WeatherContext](
  name = "Geo Agent",
  instructions = geo_instruction,
  tools = [get_geo],
  handoffs = [weather_agent]
)



instructions = f"""
{RECOMMENDED_PROMPT_PREFIX}
You are a weather forecast agent. You try to understand the user's prompt in any language.
You are not allowed to solve the task yourself. You only delegate (handoff) tasks to Geo Agent or Weather Agent.

Your workflow is as follows:
1. Extract or ask for the city name if it's not already provided.
2. Handoff the city name to the Geo Agent to get the lat and lon.
3. Other wise, handoff to Weather Agent if the lat and lon are avaiable.

NEVER stop at step 2. You MUST alwasy proceed to step 3 once the lat and lon are available.
ALWAYS pass the lat and lon to weather_agent as input when the lat and lon are available.
"""


triage_agent = Agent[WeatherContext](
  name = "Forecast Agent",
  instructions = instructions,
  handoffs= [geo_agent, weather_agent]
)

geo_agent.handoffs.append(triage_agent)
weather_agent.handoffs.append(triage_agent)


async def main():
    current_agent: Agent[WeatherContext] = triage_agent
    input_items = []
    done = True
    context = WeatherContext()

    while done:
        user_input = input("😎 Enter your message: ")
        if user_input == 'exit':
           done = False
           continue
        
        input_items.append({"content": user_input, "role": "user"})
        result = await Runner.run(current_agent, input_items, context=context)

        for new_item in result.new_items:
          agent_name = new_item.agent.name
          if isinstance(new_item, MessageOutputItem):
              print(f"🤖 {agent_name}: {ItemHelpers.text_message_output(new_item)}")
          elif isinstance(new_item, HandoffOutputItem):
              print(
                  f"👉 Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
              )
          elif isinstance(new_item, ToolCallItem):
              print(f"🤖 {agent_name}: Calling a tool")
          elif isinstance(new_item, ToolCallOutputItem):
              print(f"🤖 {agent_name}: Tool call output: {new_item.output}")
          else:
              print(f"🤖 {agent_name}: Skipping item: {new_item.__class__.__name__}")

        input_items = result.to_input_list()
        current_agent = result.last_agent
            


if __name__ == "__main__":
    asyncio.run(main())