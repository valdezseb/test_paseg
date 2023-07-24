# CODE
import streamlit as st
#from PyPDF2 import PdfReader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import openai
import os
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma, Pinecone
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
st.set_page_config(page_title="PASEG Genie ", page_icon=":coffee:", layout="wide")
# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
index_name = 'dbpaseg'

os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#openai_api_key = st.text_input("Enter OpenAI API Key:")
#os.environ['OPENAI_API_KEY'] = openai_api_key

# Define username and password
username = "ppca"
password = "65326"

# Define Streamlit app
#st.set_page_config(page_title="PASEG Genie // buy me a coffee", page_icon=":coffee:")

#st.title("Log in your Credentials")

# Prompt user for login credentials
#login_username = st.text_input("Username:")
#login_password = st.text_input("Password:", type="password")

# Prompt user for OpenAI API key
#openai_api_key = st.text_input("Enter your OpenAI API key:")

#if login_username == username and login_password == password:
#    st.success("Login successful!")
    
# Create a function to load embeddings and Pinecone client
@st.cache_resource  # Allow output mutation for the Pinecone client
def load_embeddings():
    return embeddings = HuggingFaceEmbeddings()

embeddings = load_embeddings()

@st.cache_data
def load_pinecone(embeddings):
    return docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Load the Pinecone client using st.cache
docsearch = load_pinecone(embeddings)


# Create the Chat and RetrievalQA objects
chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)
qachain = load_qa_chain(chat, chain_type='stuff')
qa = RetrievalQA(combine_documents_chain=qachain, retriever=docsearch.as_retriever())

condition1 = '\n [organize information: organize text so its easy to read, and bullet points when needed.] \n [tone and voice style: clear sentences, avoid use of complex sentences]'


st.title("PASEG Genie // for education purpose :coffee:")
st.markdown("*Chatbot for Planning and Schedule Excellence Guide*", unsafe_allow_html=True)
st.markdown("---")
# Let the user input a query
query = st.text_input("Enter your query:")

# Run the QA system and display the result using Streamlit
if query:
    result = qa.run(query + '\n' + condition1)
    st.write(result)



