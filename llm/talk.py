### Question Answering using LLM
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate

base_url = "http://localhost:11434"
model = 'llama3.2:3b'

llm = ChatOllama(base_url=base_url, model=model)

prompt = """
You are an AI assistant answering questions based ONLY on the provided context.
- Analyze the question and context. 
- Use three sentences maximum and keep the answer concise or brief.
- Answer with well-structured sentences.
- If the question is not in the context, reply: "I don't know the answer".
- **Do not** make assumptions.

Question: {question}  
Context: {context}  
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt)

def export_llm():
    return llm, prompt
# def ask_llm(context, question):
#     return qna_chain.invoke({'context': context, 'question': question})