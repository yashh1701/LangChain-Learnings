from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma world's top cricketer who known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

#adding index to the document embeddings for later use and then calculating the cosine similarity between the query embedding and document embeddings to get the similarity score for each document with the query
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# sort  on the basis of similarity score and get the index of the most similar document and its score with the query
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1] 

print(query)
print(documents[index])
print("similarity score is:", score)