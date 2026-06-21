import random
class NakliLLM:
    def __init__(self):
        print('LLM created')
    
    def predict(self,prompt):
        response_list=[
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response':random.choice(response_list)}

llm=NakliLLM()
llm.predict("what is the capital of india")