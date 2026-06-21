from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]
result=model.invoke(messages)
messages.append(AIMessage(result.content))
print(messages)
