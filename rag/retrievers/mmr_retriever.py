from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
vector_store=FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)
retriver=vector_store.as_retriever(
    search_type='mmr', # this enables mmr
    search_kwargs={"k":3,"lambda_multi":0.5} # k=top result,lambda-multi=relevance diversity balance
)
query='What is lanchain?'
result=retriver.invoke(query)
for i,doc in enumerate(result):
    print(f"\n---Result {i+1}---")
    print(doc.page_content)