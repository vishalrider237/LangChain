from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)  # type: ignore
model=ChatHuggingFace(llm=llm)
prompt1=PromptTemplate(template='Write a joke about {topic}',input_variables=['topic'])
parser=StrOutputParser()
prompt2=PromptTemplate(template='Explain the following joke - {text}',input_variables=['text'])
joke_generator_chain=RunnableSequence(prompt1,model,parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2 , model ,parser)
})
final_chain=RunnableSequence(joke_generator_chain,parallel_chain)
result=final_chain.invoke({'topic':'cricket'})
print(result)
    