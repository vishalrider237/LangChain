from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json
import requests
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)

#tool create
@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    """This function fetches the currency conversion factor between a given base currency and target currency"""
    url=f"https://v6.exchangerate-api.com/v6/b5984c1cfcac143c0f4ea7d3/pair/{base_currency}/{target_currency}"
    response=requests.get(url)
    return response.json()
result=get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})

@tool
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])-> float: # it means llm donot fill this argument,you can set the value
    """Given a  currency conversion rate this function calculates the target  currency value from a given base currency value"""
    return base_currency_value*conversion_rate

conversion_result=convert.invoke({'base_currency_value':10,'conversion_rate':94.12})

#binding tool
llm_with_tools=model.bind_tools([get_conversion_factor,convert])

message=[HumanMessage('What is the conversion factor between USD and INR ,and based on that can you convert 10 USD to INR')]
AI_message=llm_with_tools.invoke(message)
message.append(AI_message)
for tool_call in AI_message.tool_calls:
    #execute te first tool and get the value of conversion rate 
    if tool_call['name']=='get_conversion_factor':
        tool_message1=get_conversion_factor.invoke(tool_call)
        #fetch this conversion rate and append this message in message list
        conversion_rate=json.loads(tool_message1.content)['conversion_rate']
        message.append(tool_message1)
    
        
    # i have to execute the second tool using the conversion rate from tool1
    if tool_call['name']=='convert':
        # manually we need to inject the conversion rate argument
        tool_call['args']['conversion_rate']=conversion_rate
        tool_message2=convert.invoke(tool_call)
        message.append(tool_message2)
final_result=llm_with_tools.invoke(message).content
print(final_result)
    
