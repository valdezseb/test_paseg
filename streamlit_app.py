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



# Prompt user for OpenAI API key
openapikey = st.text_input("Enter your OpenAI API key:")


os.environ['OPENAI_API_KEY'] = openapikey





embeddings = HuggingFaceEmbeddings()

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)

qachain = load_qa_chain(chat, chain_type='stuff')

qa = RetrievalQA(combine_documents_chain=qachain, retriever=docsearch.as_retriever())

condition1 = '\n [organize information: organize text so its easy to read, and bullet points when needed.] \n [tone and voice style: clear sentences, avoid use of complex sentences]'

# INTEGRATE STREAMLIT TO THE FOLLOWING CODE, SO USER CAN INPUT THE QUERY, AND RESULT SHOULD A APPEAR IN A TEXT BOX USING STREAMLIT



st.title("Question Answering System")



query = st.text_input("Enter your query:")

if st.button("Get answer"):
    q = query + '\n' + condition1
    result = qa.run(q)
    st.write(result)
