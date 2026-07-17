from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

# Define a Pydantic model to represent the feedback sentiment
class Feedback(BaseModel):
    # Define the sentiment field with a description and a type of Literal to restrict the values to 'positive' or 'negative'
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Create a PydanticOutputParser using the defined Pydantic model class to parse the output of the model into a structured format
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

# Create a chain of the prompt template, model, and parser to get the final output. The chain will take the input variable 'feedback', pass it through the prompt template, then through the model, and finally through the parser to get the structured output.
classifier_chain = prompt1 | model | parser2

# Create two PromptTemplates for generating responses to positive and negative feedback, respectively. Each template takes the feedback as input and instructs the model to generate an appropriate response.
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Create a PromptTemplate for generating responses to negative feedback. The template takes the feedback as input and instructs the model to generate an appropriate response.
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Create a RunnableBranch to handle the branching logic based on the sentiment of the feedback. The branch will take the output of the classifier_chain and check the sentiment field. If the sentiment is 'positive', it will invoke prompt2 | model | parser, and if the sentiment is 'negative', it will invoke prompt3 | model | parser. If neither condition is met, it will return a default message.
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

# Create a chain of the classifier_chain and branch_chain to get the final output. The chain will first classify the sentiment of the feedback and then branch to the appropriate response based on the sentiment.
chain = classifier_chain | branch_chain

# Invoke the chain with a sample feedback
print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# print the graph of the chain to visualize the flow of data through the prompts, models, and parsers
chain.get_graph().print_ascii()