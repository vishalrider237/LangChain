from dummy_llm import NakliLLM
class NakliPromptTemplate:
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables
    def format(self,input_dict):
        return self.template.format(**input_dict)
    
template=NakliPromptTemplate(
    template='Write a poem about {topic}',
    input_variables=['topic']
)
# it will behave same as prompttemplate class
prompt=template.format({'topic':"india"})
llm=NakliLLM()
result=llm.predict(prompt)
print(result)