from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# LOAD PDF
pdf_path = r"C:\Users\alka2\OneDrive\Desktop\FaissRAG\myRAGpdf.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# SPLIT TEXT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=30
)

docs = text_splitter.split_documents(documents)

# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# VECTOR STORE
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

retriever = vectorstore.as_retriever()

# OLLAMA MODEL
llm = ChatOllama(
    model="llama3"
)

# RAG CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# QUERY
response = qa_chain.invoke({
    "query": "Give me the gist of RAG in 3 sentences"
})


print(response["result"])