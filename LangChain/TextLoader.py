# python -m pip install langchain-community
from langchain_community.document_loaders import TextLoader #text file ko read/load krna
from langchain_openai import ChatOpenAI #llm model ko langChain mei use krna 
from langchain_core.output_parsers import StrOutputParser #llm response ko message formate mei lana 
from langchain_core.prompts import PromptTemplate #dynamic prompt banane ke liye yani prompt mei variable rakh skte h eg {poem}
from dotenv import load_dotenv 

load_dotenv()

model = ChatOpenAI() #created the model

#created the prompt
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser() #ai respode ko string mei convert krna

loader = TextLoader('cricket.txt', encoding='utf-8') #txt load kiya

docs = loader.load() #txt read krna

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))
