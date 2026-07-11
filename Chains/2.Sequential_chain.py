from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# creating 1st prompt to generate a detailed report on the given topic
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# creating 2nd prompt to generate a 5-pointer summary from the detailed report
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

# creating a parser to convert the output of the model into a string
parser = StrOutputParser()

# creating a chain of prompts, models and parsers to get the final output
chain = prompt1 | model | parser | prompt2 | model | parser

# invoking the chain with the input variable 'topic' and printing the final output
result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

# printing the graph of the chain to visualize the flow of data through the prompts, models, and parsers
chain.get_graph().print_ascii()