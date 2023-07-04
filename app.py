import streamlit as st
import os
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title('PDFquery')
    st.markdown('''
    A powerful app using Langchain and 
    OpenAI for advanced language processing, 
    with Streamlit as the intuitive frontend. 
    Langchain provides natural language understanding 
    and generation, while OpenAI delivers 
    cutting-edge results. Streamlit simplifies 
    the user interface, enabling seamless 
    interaction with the app's language processing 
    features. Experience the synergy of these 
    technologies in one cohesive application.
    ''')
    add_vertical_space(5)
    st.write('-----------------------')

def main():
    load_dotenv()
    st.header("pdfQuery ChatApp")

    pdf = st.file_uploader('Upload PDF file',type='pdf')
    
    text=""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 150,
            length_function = len,
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        if os.path.exists(f'{store_name}.pkl'):
            with open (f'{store_name}.pkl','rb') as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f'{store_name}.pkl','wb') as f:
                pickle.dump(VectorStore,f)
    
        query = st.text_input("Query related to PDF")
        if(query):
            docs = VectorStore.similarity_search(query=query,k=2)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm,chain_type='stuff')
            with get_openai_callback() as cost:
                res = chain.run(input_documents=docs,question=query)
                print(cost)
                st.write(res)

if __name__ == '__main__':
    main()