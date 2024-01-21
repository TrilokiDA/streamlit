# Created by triloki at 01-01-2024

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

# sidebar contents
with st.sidebar:
    st.title('LLM Chat App :robot_face:')
    st.markdown('''
    ## About
    This app is an LLM_Powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')
    add_vertical_space(5)
    st.write('Made with :heart: by [Promt Engineer](https://github.com/TrilokiDA)')


def main():
    st.header("Chat with PDF :memo:")

    # Upload a pdf file
    pdf = st.file_uploader("Upload your PDF file", type='pdf')
    # st.write(pdf.name)
    # st.write(pdf_reader)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        # create embedding

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            st.write("Embedding loaded from Disk")
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            # st.write("Embedding computation completed")

        query = st.text_input("Ask questions from your PDF file")
        st.write(query)

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
