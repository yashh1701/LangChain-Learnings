from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)
 
# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 1 line summary on the following text. /n {text}',
    input_variables=['text']
)

#creating a parser to convert the output of the model into a string
parser = StrOutputParser()

#creating a chain of prompts, models and parsers to get the final output
chain = template1 | model | parser | template2 | model | parser

#invoking the chain with the input variable 'topic' and printing the final output
result = chain.invoke({'topic':'black hole'})

print(result)
