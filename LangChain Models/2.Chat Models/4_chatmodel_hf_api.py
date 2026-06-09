from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint  # for creating a llm instance from HuggingFace API endpoint and pass it to the ChatHuggingFace class
import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(env_path)

# print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", # HuggingFace model repository ID (like github repo, but for HuggingFace models)
    task="text-generation" # which task to perform, it can be text-generation, text2text-generation, etc. depending on the model you are using
)

model = ChatHuggingFace(llm=llm)  # passing the llm instance to the ChatHuggingFace class

result = model.invoke("Tell me about Virat Kohli and Rohit Sharma bonding. How they are share there bond on and off the field?")  # invoking the model with a prompt

print(result.content)