from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)

#create tool
@tool
def multiply(a:int,b:int)->int:
    """Given 2 numbers a and b this tool returns their product"""
    return a*b

#tool bind
llm_with_tools=model.bind_tools([multiply])

#tool execution
query=HumanMessage('can you multiply 3 with 10')
messages=[query]
result=llm_with_tools.invoke(messages)
messages.append(result)
final_result=multiply.invoke(result.tool_calls[0]) # it will give tool message
messages.append(final_result)
tool_result=llm_with_tools.invoke(messages)
print(tool_result)

