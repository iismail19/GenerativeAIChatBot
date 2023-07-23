# Local Chatbot without openai embeddings or openai LLM

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import re
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<ENTER_KEY_HERE>"

# load pdf document and get chunks
def get_pdf_text(filePath):
    loader = PyPDFLoader(filePath)
    doc = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap  = 0,
        length_function = len,
    )
    chunks = text_splitter.split_documents(doc)
    return chunks

# get embeddings - open source embedding (Many available depending on application usage)
def get_embeddings():
    return HuggingFaceEmbeddings()

# Create vectorstores - will now have our embedded documents
def get_vector_stores(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)

import textwrap

def wrap_text_preserve_newlines(text, width=200):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [re.sub(' +', ' ', textwrap.fill(line, width=width)) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def large_language_model():
    return HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":.05, "max_length":1024})


def main():
    file_path = 'src/resources/intern.pdf'
    documents = get_pdf_text(file_path)

    embeddings = get_embeddings()

    vector_stores = get_vector_stores(documents, embeddings)

    # Test and ask qeustions - this is based on similarity
    query = "How many hours are needed to complete an internship?"
    docs = vector_stores.similarity_search(query)
    
    #print(wrap_text_preserve_newlines(str(docs[0].page_content)))

    # Integrate it as part of a Q and A chain
    llm = large_language_model()
    chain = load_qa_chain(llm, chain_type="stuff")
    chain.run(input_documents=docs, question=query)
    print(wrap_text_preserve_newlines(str(docs[0].page_content)))

if __name__ == "__main__":
    main()