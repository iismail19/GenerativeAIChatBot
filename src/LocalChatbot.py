# Local Chatbot without openai embeddings or openai LLM

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import re

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<ENTER_KEY_HERE>"

# load pdf document and get chunks
def get_pdf_text(filePath):
    loader = PyPDFLoader(filePath)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(doc)
    return chunks

# get embeddings - open source embedding (Many available depending on application usage)
def get_embeddings():
    return HuggingFaceEmbeddings()

# Create vectorstores - will now have our embedded documents
def get_vector_stores(documents, embeddings):
    db = FAISS.from_documents(documents, embeddings)
    return db

import textwrap

def wrap_text_preserve_newlines(text, width=200):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [re.sub(' +', ' ', textwrap.fill(line, width=width)) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def main():
    file_path = 'src/resources/cleanCode.pdf'
    documents = get_pdf_text(file_path)

    embeddings = get_embeddings()

    vector_stores = get_vector_stores(documents, embeddings)

    # Test and ask qeustions
    query = "What is the purpose of writing clean code?"
    results = vector_stores.similarity_search(query)
    
    print(wrap_text_preserve_newlines(str(results[0].page_content)))



if __name__ == "__main__":
    main()