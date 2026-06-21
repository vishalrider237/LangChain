from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.prompts import PromptTemplate
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',input_variables=['poem']
)
parser=StrOutputParser()
loader=TextLoader('cricket.txt',encoding='utf-8')
document=loader.load()
chain=prompt | model | parser
result=chain.invoke({'poem':document[0].page_content})
print(result)