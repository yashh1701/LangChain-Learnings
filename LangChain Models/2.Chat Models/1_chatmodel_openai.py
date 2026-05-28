from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.15)

result = model.invoke("Write a code in java to print prime numbers between 1 to 100")

print(result.content)