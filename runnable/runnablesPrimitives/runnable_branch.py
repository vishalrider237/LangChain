from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda ,RunnablePassthrough,RunnableBranch
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)
prompt1=PromptTemplate(template='Write a detailed report on  {topic}',input_variables=['topic'])
parser=StrOutputParser()
prompt2=PromptTemplate(template='Summerize the following text \n {text}',input_variables=['text'])
report_gen_chain=prompt1|model|parser
branch_chain=RunnableBranch(
    (lambda x:len(x.split())>300,prompt2|model |parser),
    RunnablePassthrough()
)
final_chain=report_gen_chain |branch_chain
result=final_chain.invoke({'topic':'Russia vs Ukrain'})
print(result)