from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

# This example demonstrates the use of RunnableBranch to conditionally execute different chains based on the length of the generated report. If the report exceeds 300 words, it will be summarized; otherwise, it will be passed through as is.
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# This prompt is used to summarize the report if it exceeds 300 words.
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# The ChatOpenAI model is used to generate text based on the provided prompts.
model = ChatOpenAI()

# The output parser is used to parse the output of the model into a string format.
parser = StrOutputParser()

# The report generation chain generates a detailed report based on the provided topic.
report_gen_chain = prompt1 | model | parser

# The branch_chain checks the length of the generated report. If it exceeds 300 words, it will summarize the report using prompt2; otherwise, it will pass the report through unchanged.
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

# The final chain combines the report generation and the conditional branching into a single sequence. It first generates the report and then decides whether to summarize it or not based on its length.
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Invoke the final chain with a topic. The final chain will first generate a detailed report on the given topic and then check its length. If the report exceeds 300 words, it will summarize it; otherwise, it will return the report as is.
print(final_chain.invoke({'topic':'Russia vs Ukraine'}))


