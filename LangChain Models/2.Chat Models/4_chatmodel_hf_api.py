from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint  # create a llm instance from HuggingFace API endpoint and pass it to the ChatHuggingFace class
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)  # passing the llm instance to the ChatHuggingFace class

result = model.invoke("What is the capital of India")

print(result.content)