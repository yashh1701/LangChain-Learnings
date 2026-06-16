from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history = [
    SystemMessage(content='You are a helpful AI assistant') # system message to set the context for the conversation
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))  # convert user input to HumanMessage and append to chat_history
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) # convert model output to AIMessage and append to chat_history
    print("AI: ",result.content)

print(chat_history)