from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
) # type: ignore

llm = ChatHuggingFace(llm=llm_endpoint)
#Step1 Indexing
video_id='Gfr50f6ZBvo'
try:
    ytt_api=YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id,languages=["en"])
    text = " ".join(snippet.text for snippet in transcript)
except TranscriptsDisabled:
    print('No caption availble for this video')

# text splitter
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents([text])

#Embeding generation and storing in vector store
vector_store=FAISS.from_documents(chunks,embeddings)
ids=vector_store.index_to_docstore_id
last_chunk=vector_store.get_by_ids(['d40aa4a7-aed0-470d-99bf-efe19161871b'])

#Step2 -Retriever
retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={"k":4})

result=retriever.invoke("What is deepmind?")

#Step3- Augmentation
prompt=PromptTemplate(
    template="""
    You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,input_variables=['context','question']
    
)
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt=prompt.invoke({'context':context_text,'question':question})

#step4 - Generation
answer=llm.invoke(final_prompt)

### Building a Chain
def format_docs(retrieved_doc):
    context_text="\n\n".join(doc.page_content for doc in retrieved_doc)
    return context_text
parallel_chain=RunnableParallel({
    'context':retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})
parser=StrOutputParser()
main_chain=parallel_chain | prompt | llm | parser
print(main_chain.invoke('Can you summerize the video?'))