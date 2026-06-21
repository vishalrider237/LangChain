from langchain_core.tools import StructuredTool
from pydantic import BaseModel,Field

class MultiplyInput(BaseModel):
    a:int=Field(description='the first number to add',required=True)
    b:int=Field(description='the second number to add',required=True)
    
def multiply_func(a:int,b:int)-> int :
    return a*b
multiply_tool=StructuredTool.from_function(
    func=multiply_func,
    name='multiply',
    description='Multiply two numbers',
    args_schema=MultiplyInput
)
result=multiply_tool.invoke({'a':3,'b':5})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
    