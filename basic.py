from agents import Agent, Runner, RunConfig, handoff, AsyncOpenAI, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled, set_default_openai_api
from agents.tool import function_tool
from pydantic import BaseModel

import asyncio
import os

# Set Up Azure model, see https://openai.github.io/openai-agents-python/models/ for more information
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")


print(AZURE_OPENAI_API_VERSION)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

set_tracing_disabled(disabled=True)

external_client = AsyncOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
    default_headers={"api-key": AZURE_OPENAI_API_KEY},
    default_query={"api-version": AZURE_OPENAI_API_VERSION}
)

set_default_openai_client(external_client, use_for_tracing=False)
set_default_openai_api("chat_completions")

# Define a structured answer
class StructuredAnswer(BaseModel):
  decision: str
  reasoning: str

# Define a fake tool
@function_tool
def get_forecast(city: str) -> str:
  return f"The forecast in {city} is sunny."

# Define agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You only speak Spanish.",
    tools=[get_forecast],
    model=OpenAIChatCompletionsModel(
        model=AZURE_OPENAI_MODEL,
        openai_client=external_client,
    ),
)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English.",
    tools=[get_forecast],
    model=OpenAIChatCompletionsModel(
        model=AZURE_OPENAI_MODEL,
        openai_client=external_client,
    ),
)

triage_agent = Agent(
  name="triage_agent",
  instructions="Switch to Spanish if user uses Spanish, otherwise English.",
  handoffs=[handoff(spanish_agent), handoff(english_agent)],
  output_type=StructuredAnswer,
  model=OpenAIChatCompletionsModel(
    model=AZURE_OPENAI_MODEL,
    openai_client=external_client,
 ),
)

# Kick off process
config = RunConfig()

async def main():
    user_query = "Hola, ¿qué tiempo hace en Madrid?" # Spanish for What's the weather like in Madrid?
    print(f"😎 : {user_query}")
    result = await Runner.run(
        triage_agent, 
        user_query, 
        run_config=config,
    )
    print(f"🤖 : {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())