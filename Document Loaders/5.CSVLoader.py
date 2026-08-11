from langchain_community.document_loaders import CSVLoader

# a simple example of loading a CSV file and printing the number of documents and the second document
loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])