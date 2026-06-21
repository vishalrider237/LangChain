class NakliLLMChain:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt
    
    def run(self,input_dict):
        final_prompt=self.prompt.format(input_dict)
        result=self.llm.predict(final_prompt)
        return result['response']