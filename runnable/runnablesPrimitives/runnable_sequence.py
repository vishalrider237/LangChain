from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)
prompt1=PromptTemplate(template='Write a joke about {topic}',input_variables=['topic'])
prompt2=PromptTemplate(template='Explain the following joke- {text}',input_variables=['text'])
parser=StrOutputParser()
chain =RunnableSequence(prompt1 ,model,parser,prompt2,model,parser)
result=chain.invoke({'topic':'Ai'})
print(result)