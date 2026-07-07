from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model definition using HuggingFaceEndpoint and specify the repo_id and task for the model you want to use
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# Create a ChatHuggingFace model using the defined llm
model = ChatHuggingFace(llm=llm)

# Create a PydanticOutputParser to parse the output of the model into a Pydantic model by defining a Pydantic model class with the required fields and their types. 
# The parser will use this model to validate and parse the output of the model into a structured format.
class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

# Create a PydanticOutputParser using the defined Pydantic model class
parser = PydanticOutputParser(pydantic_object=Person)

# Create a PromptTemplate that includes the format instructions from the parser. The format instructions will be included in the prompt so that the model knows to output in the specified format.
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# Create a chain of the prompt template, model, and parser to get the final output. The chain will take the input variable 'place', pass it through the prompt template, then through the model, and finally through the parser to get the structured output.
chain = template | model | parser

# Invoking the chain with the input variable 'place' and printing the final output. The final output will be a Pydantic model instance with the name, age, and city of a fictional person from the specified place.
final_result = chain.invoke({'place':'sri lankan'})

print(final_result)