from langchain_core.tools import BaseTool
from typing import Type
from pydantic import Field,BaseModel
#arg schema using pydantic
class MultiplyInput(BaseModel):
    a:int=Field(description='this first number to add')
    b:int=Field(description='The second number to add')

class MultiplyTool(BaseTool):
    name:str='multiply'
    description:str='Multiply two number'
    args_schema:Type[BaseModel]=MultiplyInput
    def _run(self,a:int,b:int)->int:
        return a*b

multiply_tool=MultiplyTool()
result=multiply_tool.invoke({'a':3,'b':5})
print(result)