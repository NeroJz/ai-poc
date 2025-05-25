from agents import Agent, handoff, RunContextWrapper, Runner, AsyncOpenAI, set_default_openai_client, set_default_openai_api
from agents.tool import function_tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv
import os
import asyncio
import gradio as gr


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

OPEN_WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAPI_KEY = os.getenv('OPENAI_API_KEY')

print(AZURE_OPENAI_API_VERSION)

external_client = AsyncOpenAI(
   api_key = OPENAPI_KEY
    # api_key=AZURE_OPENAI_API_KEY,
    # base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
    # default_headers={"api-key": AZURE_OPENAI_API_KEY},
    # default_query={"api-version": AZURE_OPENAI_API_VERSION}
)


set_default_openai_client(external_client, use_for_tracing=False)
# set_default_openai_api("chat_completions")


async def transcribe(audio_path):
  with open(audio_path, 'rb') as audio_file:
    transcript = await external_client.audio.transcriptions.create(
      model="whisper-1",
      # model = "gpt-4o-mini-transcribe",
      file=(audio_path, audio_file, "audio/wav"),
      response_format="text"
    )
  print(f"Transcription: {transcript}")
  return transcript


async def transcribe_and_route(audio_path):
  text = await transcribe(audio_path)
  return text


iface = gr.Interface(
    fn=transcribe_and_route,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=["text"],
    title="Voice Intent Agent",
    description="Uses Whisper for transcription and Agent SDK for intent recognition."
)

if __name__ == "__main__":
    iface.launch()