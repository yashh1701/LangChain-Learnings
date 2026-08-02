from langchain_core.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# The prompt template is used to generate a summary for the provided poem. It takes the poem as input and formats it into a prompt that can be processed by the model.
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

# The TextLoader is used to load the content of the 'cricket.txt' file. It reads the file and prepares it
loader = TextLoader('cricket.txt', encoding='utf-8')

# The load method of the TextLoader is called to read the content of the file and return it as a list of Document objects. Each Document object contains the text content and metadata associated with the loaded document.
docs = loader.load()

# The type of the loaded documents is printed to verify that they are indeed Document objects. This helps in understanding the structure of the loaded data.
print(type(docs))

# The length of the loaded documents is printed to show how many Document objects were created from the file. This gives an idea of how many separate pieces of content were extracted from the file.
print(len(docs))

# The content & metadata of the first loaded document is printed to show the actual text that was read from the file. This allows for a quick inspection of the data that will be processed by the model.
print(docs[0].page_content)
print(docs[0].metadata)

# The chain is created by combining the prompt, model, and parser. The prompt generates a summary request for the poem, the model processes this request to generate a summary, and the parser formats the output into a string.
chain = prompt | model | parser

# The chain is invoked with the content of the first loaded document (the poem). The invoke method processes the input through the chain, generating a summary for the poem based on the prompt and model.
print(chain.invoke({'poem':docs[0].page_content}))
