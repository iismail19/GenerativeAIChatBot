# Imports
from langchain import VectorDBQA
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<ENTER_KEY_HERE>"

# upload a document
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_pdf_text(filePath):
#     pdfReader = PdfReader(filePath)
#     raw_text = ''
#     for i, page in enumerate(pdfReader.pages):
#         text = page.extract_text()
#         if text:
#             raw_text += text
#     return raw_text

def get_pdf_text(filePath):
    loader = PyPDFLoader(filePath)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    chunks = text_splitter.split_documents(doc)
    return chunks

# Turn document to chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
     )

    chunks = text_splitter.split_text(text)
    return chunks

# Convert to vector db

def get_vectorDb(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorDb = Chroma.from_documents(text_chunks, embeddings)
    return vectorDb

def get_conversation_chain(vectorDb, largeLangModel):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=largeLangModel,
        retriever=vectorDb.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    filePath = 'env/src/resources/cleanCode.pdf'
    pdf_text = get_pdf_text(filePath)
    # text_chunks = get_text_chunks(pdf_text)
    vectorDb = get_vectorDb(pdf_text)
    largeLangModel = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":1024})
    qa = VectorDBQA.from_chain_type(llm=largeLangModel, chain_type="stuff", vectorstore=vectorDb)
    query = "What is clean code?"
    result = qa.run(query)
    print(result)
    query2 = "What is bad code?"
    result2 = qa.run(query2)
    print(result2)

if __name__ == "__main__":
    main()