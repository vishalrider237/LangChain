from langchain_core.tools import tool

#step 1 - create a function
@tool
def multiply(a:int,b:int)->int:
    """Multiply two numbers""" # docs string ,higly recomended
    return a*b

result=multiply.invoke({"a":3,"b":5})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())
 