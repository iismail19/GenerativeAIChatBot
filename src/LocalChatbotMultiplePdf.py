# Chatbot for multipe pdfs

from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<ENTER_KEY_HERE>"


def main():
    # Load the PDF document and create a vector store from it using huggingface transformers
    pdf_folder_path = 'src/resources/'
    loaders = [UnstructuredPDFLoader(os.path.join(
        pdf_folder_path, file)) for file in os.listdir(pdf_folder_path)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=0,
            length_function=len,
        )).from_loaders(loaders)

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                         model_kwargs={"temperature": 0.5, "max_length": 1024})

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=index.vectorstore.as_retriever(),
                                        input_key="question")

    print(chain.run('How many hours does an intern need to complete?'))

    print(chain.run('What is clean code?'))


if __name__ == "__main__":
    main()
