# app/core/pipeline.py
import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

load_dotenv()
# if not os.environ.get("GROQ_API_KEY"):
#     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="../chroma_langchain_db",
)

print("Executing the template..")

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know Use 60 characters maximum and 
keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)
prompt = custom_rag_prompt

print("finished executing the template..")
def process_pdf_folder(folder_path: str) -> List[Document]:
    from langchain_community.document_loaders import PDFMinerLoader
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            loader = PDFMinerLoader(os.path.join(folder_path, filename), mode="page")
            all_docs.extend(loader.load())
    return all_docs

def split_docs(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)

def answer_question(question: str) -> str:
    retrieved_docs = vector_store.similarity_search(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs[:3])
    messages = prompt.invoke({"question": question, "context": context})
    response = llm.invoke(messages)
    return response.content
