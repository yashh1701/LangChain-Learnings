from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# structured output parser allows you to define a schema for the output of the model and parse the output into a structured format based on that schema.
# but validation is not done on the output of the model, it just parses the output into a structured format based on the schema defined.
# creating a schema for the output of the model using ResponseSchema class. The schema defines the name and description of each field in the output.
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# creating a StructuredOutputParser using the defined schema. The parser will use this schema to parse the output of the model into a structured format.
# from_response_schemas is a class method that creates a StructuredOutputParser instance from the defined schema.
# The parser will use the schema to parse the output of the model into a structured format based on the defined fields and their descriptions.
parser = StructuredOutputParser.from_response_schemas(schema)

# creating a PromptTemplate that includes the format instructions from the parser. The format instructions will be included in the prompt so that the model knows to output in the specified format.
# partial_variables is used to pass the format instructions to the prompt template so that it can be included in the prompt.
#by deafualt get_format_instructions() method of the StructuredOutputParser class returns the format instructions in json schema format,
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# creating a chain of the prompt template, model, and parser to get the final output. The chain will take the input variable 'topic', pass it through the prompt template, then through the model, and finally through the parser to get the structured output.
chain = template | model | parser

# invodke() method of the chain is used to invoke the chain with the input variable 'topic' and get the final output. 
# The final output will be a structured format based on the defined schema.
result = chain.invoke({'topic':'black hole'})

print(result)