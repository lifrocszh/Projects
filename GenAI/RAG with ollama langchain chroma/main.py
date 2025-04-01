from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from db import retriever

model  = OllamaLLM(model = 'llama3.2')

template = '''
You are an expert in answeing questions about random facts about the solar system and outer space.
You answers should be short and straight to the point.
If the answers require you to state measurements, answer in metric units. 

Here are some relevant information: {information}

Here are is the corresponding question to answer: {question}
'''
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model



while True:
    print('\n\n--------------------------------------')
    question = input('Question (type /bye to quit)')
    print('\n\n')
    if question == '/bye':
        break
    
    top_k_relevant_chunks = retriever.invoke(question)
    print('retrieved from db: \n', top_k_relevant_chunks,)
    
    result = chain.invoke({'information': top_k_relevant_chunks, 
        'question': question})
    print('\nanswer from llm: \n', result)
    