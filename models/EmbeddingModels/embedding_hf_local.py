from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
document=[
    "Delhi is the capital of india",
    "Kolkata is the capital of west bengal",
    "Paris is the capital of France"
]
vector=embedding.embed_documents(document)
print(str(vector))