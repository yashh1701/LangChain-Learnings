from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    # This will pass the joke generated from the first chain to the second chain for explanation
    'joke': RunnablePassthrough(), 
    # This will take the joke and generate an explanation for it
    'explanation': RunnableSequence(prompt2, model, parser) 
})

# The final chain combines the joke generation and explanation into a single sequence
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke the final chain with a topic
print(final_chain.invoke({'topic':'cricket'}))