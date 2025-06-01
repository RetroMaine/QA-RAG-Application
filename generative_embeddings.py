from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 

### SETTING UP PDF LOADER ###
loaders = [PyPDFLoader('data/gpu.pdf')]

docs = []

# looping all the files and add those files in the docs
for file in loaders:
    docs.extend(file.load())


### SPLITTING TEXT INTO CHUNKS ###
# we want to set up managable vectorized chunks of data for embeddings and for vector db

# chunk_size controls the size of each individual chunk in characters
    # smaller chunks allow for more focused snippets
# chunk_overlay controls how much the adjacent chunks with overlap with others
    # ensures you don't lose critical context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# function to ingest docs and spit out embeddings
embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# persistent make sure that the database is persistent and not only within our memory
vectorstore = Chroma.from_documents(docs, embeddings_function, persist_directory="./chroma_db_nccn")

print(vectorstore._collection.count()) # 52 individual embeddings from pdf
 