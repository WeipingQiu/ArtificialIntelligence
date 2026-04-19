from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap,RunnablePassthrough
from langchain.prompts import PromptTemplate
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)




# 1. Load and Split file.
loader = TextLoader("./documents/law.txt")
# 
documents = loader.load()
# print(documents)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
splits = text_splitter.split_documents(documents)
text_list = []
for i in splits:
    print(i)
    text_list.append(i)



prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""请根据检索的上下文回答问题。
    
    上下文：
    {context}

    问题：
    {query}

    答案："""
)



# embedding, and vector store

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


vectorstore = Chroma.from_documents(
                                    #文档切片
                                    splits, 
                                    #本地Embedding模型
                                    embeddings,
                                    #通过Chroma将向量存储到本地
                                    persist_directory="./chrome_db"
                                    )

print(vectorstore)