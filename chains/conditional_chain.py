from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
) # pyright: ignore[reportCallIssue]

model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Give the sentiment of the feedback')
pydanticParser=PydanticOutputParser(pydantic_object=Feedback)
prompt1=PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydanticParser.get_format_instructions()}
)
classifier_chain=prompt1 | model | pydanticParser
prompt2=PromptTemplate(
    template='Write a professional response to this positive feedback: \n {feedback}',
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template='Write a professional response to this negative feedback: \n {feedback}',
    input_variables=['feedback']
)
chain = (
    RunnablePassthrough.assign(
        classification=classifier_chain
    )
    | RunnableBranch(
        (
            lambda x: x["classification"].sentiment == "positive",
            prompt2 | model | parser
        ),
        (
            lambda x: x["classification"].sentiment == "negative",
            prompt3 | model | parser
        ),
        RunnableLambda(lambda x: "Could not find sentiment")
    )
)
result=chain.invoke({'feedback':'This is a nice phone'})
print(result)
