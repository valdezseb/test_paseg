# CODE
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import openai
import os

from langchain.vectorstores import Chroma, Pinecone
import pinecone

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
index_name = 'dbpaseg'


# Define username and password
username = "ppca"
password = "65326"

# Define Streamlit app
st.set_page_config(page_title="PASEG Genie // buy me a coffee", page_icon=":coffee:")

st.title("PASEG Genie ")

login_username = st.text_input("Username:")
login_password = st.text_input("Password:", type="password")
if st.button("Login"):
    if login_username == username and login_password == password:
        st.success("Login successful!")
        with st.form("OpenAI API key"):
            openai_api_key = st.text_input("Enter your OpenAI API key:")
            submit_button = st.form_submit_button(label="Submit")
            if submit_button:
                if openai_api_key:
                    os.environ['OPENAI_API_KEY'] = openai_api_key
                    embeddings = HuggingFaceEmbeddings()
                    docsearch = Pinecone.from_existing_index(index_name, embeddings)
                    chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)
                    qachain = load_qa_chain(chat, chain_type='stuff')
                    qa = RetrievalQA(combine_documents_chain=qachain, retriever=docsearch.as_retriever())
                    condition1 = '\n [organize information: organize text so its easy to read, and bullet points when needed.] \n [tone and voice style: clear sentences, avoid use of complex sentences]'
                    query = st.text_input("Enter your query:")
                    if st.button("Get answer"):
                        q = query + '\n' + condition1
                        result = qa.run(q)
                        st.write(result)
                else:
                    st.warning("Please enter an OpenAI API key.")
    else:
        st.error("Invalid username or password.")
