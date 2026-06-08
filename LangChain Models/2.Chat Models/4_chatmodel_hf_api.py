from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint  # create a llm instance from HuggingFace API endpoint and pass it to the ChatHuggingFace class
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # HuggingFace model repository ID (like github repo, but for HuggingFace models)
    task="text-generation" # which task to perform, it can be text-generation, text2text-generation, etc. depending on the model you are using
)

model = ChatHuggingFace(llm=llm)  # passing the llm instance to the ChatHuggingFace class

result = model.invoke("What is the capital of India")

print(result.content)