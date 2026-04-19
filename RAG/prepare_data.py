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
print(documents)