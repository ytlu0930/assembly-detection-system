from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
import os
import base64

# 強制讀取 project/.env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt_deployment = os.getenv("GPT4O_DEPLOYMENT")

print("ENV path:", env_path)
print("Endpoint:", endpoint)
print("GPT4O deployment:", gpt_deployment)
print("API key loaded:", api_key is not None)

# 建立 Azure Client
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# 呼叫 GPT
response = client.chat.completions.create(
    model=os.getenv("GPT4O_DEPLOYMENT"),
    messages=[
        {
            "role": "user",
            "content": "你好，請測試API是否成功"
        }
    ]
)

print(response.choices[0].message.content)