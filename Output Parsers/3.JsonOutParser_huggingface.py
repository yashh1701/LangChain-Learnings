from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model definition using HuggingFaceEndpoint and specify the repo_id and task for the model you want to use
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Create a ChatHuggingFace model using the defined llm
model = ChatHuggingFace(llm=llm)

# Create a JsonOutputParser to parse the output of the model into JSON format
parser = JsonOutputParser()

# Create a PromptTemplate that includes the format instructions from the parser
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()} # this will include the format instructions in the prompt so that the model knows to output in JSON format
)

# Create a chain of the prompt template, model, and parser to get the final output
chain = template | model | parser

# Invoking the chain with the input variable 'topic' and printing the final output
result = chain.invoke({'topic':'gen ai'})

print(result)
