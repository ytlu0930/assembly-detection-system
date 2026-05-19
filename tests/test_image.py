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
image_deployment = os.getenv("IMAGE_DEPLOYMENT")

print("ENV path:", env_path)
print("Endpoint:", endpoint)
print("Image deployment:", image_deployment)
print("API key loaded:", api_key is not None)

client = AzureOpenAI(
    api_version="2025-04-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key
)

result = client.images.generate(
    model=image_deployment,
    prompt="一隻可愛的黑貓坐在書桌前",
    size="1024x1024"
)

image_base64 = result.data[0].b64_json

output_path = Path(__file__).resolve().parents[1] / "output" / "output.png"
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "wb") as f:
    f.write(base64.b64decode(image_base64))

print(f"圖片生成成功：{output_path}")