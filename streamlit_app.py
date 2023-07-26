# CODE
import streamlit as st
from PyPDF2 import PdfReader
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
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components
import re
import datetime as dt
import numpy as np

#st.set_page_config(page_title="PASEG Genie ", page_icon=":coffee:")
st.set_page_config(page_title="PASEG Genie ", page_icon=":coffee:", layout="wide")

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
index_name = 'db-paseg'

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

#@st.cache_resource
#def load_embeddings_and_pinecone():
#    embeddings = HuggingFaceEmbeddings()
#    docsearch = Pinecone.from_existing_index(index_name, embeddings)
#    return docsearch

@st.cache_resource
def load_embedding():
    embeddings = HuggingFaceEmbeddings()
    return embeddings

embeddings = load_embedding()

def load_pinecone(embeddings, index_name):
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch





# Load the Pinecone client using st.cache
#docsearch = load_embeddings_and_pinecone()
# Load the Pinecone client using st.cache
docsearch = load_pinecone(embeddings, "db-paseg")
# Create the Chat and RetrievalQA objects
chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)
qachain = load_qa_chain(chat, chain_type='stuff')
qa = RetrievalQA(combine_documents_chain=qachain, retriever=docsearch.as_retriever())

condition1 = '\n [Generate Response/Text from my data.]  \n [organize information: organize text so its easy to read, and bullet points when needed.] \n [if applicable for the question response, add section: Things to Promote/Things to Avoid and Best Practices, give Examples] \n [tone and voice style: clear sentences, avoid use of complex sentences]'

st.title("PASEG Genie // for education purpose :coffee:")
#st.markdown("Donate a coffee")
st.markdown("*Chat With The Planning and Schedule Excellence Guide ver. 5.0*", unsafe_allow_html=True)
st.markdown("---")
# Let the user input a query
query = st.text_input("Enter your query:")
# Run the QA system and display the result using Streamlit
if query:
    result = qa.run(query + '\n' + condition1)
    st.write(result)

st.markdown("---")
st.markdown("Data Analysis Section")

@st.cache(suppress_st_warning=True)
def process_data(uploaded_file):
        
        columns_check = ['ID',
         'Active',
         'Task_Mode',
         'Task_Name',
         'Duration',
         'Start_Date',
         'Finish_Date',
         'Predecessors',
         'Resource_Names',
         'Actual_Start',
         'Actual_Finish',
         'Total_Slack',
         'Successors',
         'Unique_ID',
         'Outline_Level',
         'Constraint_Date',
         'Constraint_Type',
         'Early_Start',
         'Early_Finish',
         'Late_Start',
         'Late_Finish',
         'Notes']
    
       
        
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        try:
            if df.columns.tolist() != columns_check:
                raise ValueError("DataFrame columns do not match expected columns")
        except ValueError as e:
            st.error("Columns not match template!!")
            st.warning("Check your dataframe: " + str(e))
        else:
            st.balloons()
            st.success("Excel File was read Successfully!")
            st.toast("Data begins to process, just a second...")
            # specify the format for the date strings
            date_format = '%Y-%m-%d %H:%M:%S'
            date_format_finish = '%B %d, %Y %I:%M %p'
            date_format_as = '%m/%d/%y'
            df['Finish_Date'] = pd.to_datetime(df['Finish_Date'], format=date_format_finish)
            df['Actual_Start'] = pd.to_datetime(df['Actual_Start'], format=date_format_as)
            df['Actual_Finish'] = pd.to_datetime(df['Actual_Finish'], format=date_format_as)
            df['Start_Date'] = pd.to_datetime(df['Start_Date'], format=date_format_finish)
            
            # define a custom function to convert duration strings to timedelta
            # check the dtype of the Duration column
            if df['Duration'].dtype == 'datetime64[ns]':
                # convert the datetime values to timedelta format
                df['Duration'] = (pd.to_datetime(df['Duration']) - pd.to_datetime(df['Duration']).min()).astype('timedelta64[ns]')
            else:
                # convert the Duration column to timedelta format
                df['Duration'] = pd.to_timedelta(df['Duration'], errors='coerce')
            #df['Duration'] = df['Duration'].apply(duration_to_timedelta)
    
    
        
            # Split the Predecessors column by comma and join the values that contain a special character or word
            df['task_dependency'] = df['Predecessors'].apply(lambda x: ','.join([y for y in str(x).split(',') if any(c.isalpha() for c in y)]))
            
            def clean_string(s):
                # Split the string by ","
                s = s.split(",")
            
                # Remove letters and special characters from each substring
                s = [re.sub(r"[^\d+]", "", x) for x in s]
            
                # Remove the number followed by "+" in each substring
                s = [re.sub(r"\+\d+", "", x) for x in s]
            
                # Convert each non-empty substring to an integer
                s = [int(x) for x in s if x]
            
                return s
            
            # Apply the function to each row of the "Task Dependencies" column and create a new column "task_dependency"
            df["task_dependency_ids"] = df["task_dependency"].apply(clean_string)
            
            def get_max_dependency_dates(df):
                # Step 1: Create a dictionary that maps each unique ID to its start and finish dates
                id_to_dates = dict(zip(df["Unique_ID"], zip(df["Start_Date"], df["Finish_Date"])))
            
                # Step 2: Iterate over each row of the DataFrame and find the max dependency start and finish dates
                max_dependency_starts = []
                max_dependency_finishes = []
                for i, row in df.iterrows():
                    # Get the task_dependency value
                    dependency_str = row["task_dependency"]
            
                    # Check if the task_dependency contains "SS" or "SF"
                    if "SS" in dependency_str or "SF" in dependency_str:
                        # Get the task_dependency_ids value
                        dependency_ids = row["task_dependency_ids"]
            
                        # Find the start and finish dates for each dependency ID using the id_to_dates dictionary
                        dependency_dates = [id_to_dates[id] for id in dependency_ids if id in id_to_dates]
            
                        # Calculate the max start and finish dates from the dependency IDs
                        if dependency_dates:
                            max_dependency_start = min([d[0] for d in dependency_dates])
                            max_dependency_finish = max([d[1] for d in dependency_dates])
                        else:
                            max_dependency_start = row["Start_Date"]
                            max_dependency_finish = row["Finish_Date"]
                    else:
                        max_dependency_start = ""
                        max_dependency_finish = ""
            
                    # Store the max start and finish dates in new columns of the DataFrame
                    max_dependency_starts.append(max_dependency_start)
                    max_dependency_finishes.append(max_dependency_finish)
            
                df["max_dependency_start"] = max_dependency_starts
                #df["max_dependency_finish"] = max_dependency_finishes
            
                return df
            
            df = get_max_dependency_dates(df)
            
            
            # Step 1: Create a dictionary that maps each unique ID to its finish date
            id_to_finish = dict(zip(df["Unique_ID"], df["Finish_Date"]))
            
            # Step 2: Iterate over each row of the DataFrame and find the max dependency finish date
            max_dependency_finishes = []
            for i, row in df.iterrows():
                # Check if the task_dependency contains "FF"
                if "FF" in row["task_dependency"]:
                    # Get the task_dependency_ids value
                    dependency_ids = row["task_dependency_ids"]
            
                    # Find the finish date for each dependency ID using the id_to_finish dictionary
                    dependency_finishes = [id_to_finish[id] for id in dependency_ids if id in id_to_finish]
            
                    # Calculate the max finish date from the dependency IDs
                    if dependency_finishes:
                        max_dependency_finish = max(dependency_finishes)
                    else:
                        max_dependency_finish = row["Start_Date"]
                else:
                    max_dependency_finish = ""
            
                # Store the max finish date in a new column of the DataFrame
                max_dependency_finishes.append(max_dependency_finish)
            
            df["max_dependency_finish"] = max_dependency_finishes
            
            # Convert the "max_dependency_start" and "max_dependency_finish" columns to datetime format
            df["max_dependency_start"] = pd.to_datetime(df["max_dependency_start"]).dt.date
            df["max_dependency_finish"] = pd.to_datetime(df["max_dependency_finish"]).dt.date
            
            # Remove the brackets from the "a" column
            df["task_dependency_ids"] = df["task_dependency_ids"].astype(str).str.replace("[", "").str.replace("]", "");
            
            df["new_task_dependency"] = np.where(df["task_dependency"] == "", "No Dependency",
                                                  np.where(df["task_dependency"] == "nan", "No Predecessors",
                                                           df["task_dependency"]))
            
            # Define a regular expression to extract only numbers and commas
            pattern = r'\b(\d+)(?:,|$)'
            
            # Apply the regular expression to the "predecessors" column and extract all matches
            df['Predecessors_clean'] = df['Predecessors'].str.extractall(pattern).groupby(level=0).agg(','.join)
            
            # Define a function to compute the task names for each row
            def compute_predecessor_names(row):
                predecessors = row['Predecessors_clean']
                if pd.isna(predecessors):
                    return ''
                else:
                    predecessors = set(map(int, predecessors.split(',')))
                    predecessor_names = list(df.loc[df['Unique_ID'].isin(predecessors), 'Task_Name'])
                    return ','.join(predecessor_names)
            
            # Apply the function to each row and create a new column
            df['Critical_Path'] = df.apply(compute_predecessor_names, axis=1)
            
            #time stamp for current time
            now = dt.datetime.now()
            
            #Prepare datetime objects for further data analysis
            df['Start_Date'] = pd.to_datetime(df['Start_Date'].apply(lambda x: x.strftime('%b %d, %Y')))
            df['Finish_Date'] = pd.to_datetime(df['Finish_Date'].apply(lambda x: x.strftime('%b %d, %Y')))
            
            #Prepare datetime objects for further data analysis
            df['max_dependency_start'] = df['max_dependency_start'].apply(lambda x: x.strftime('%b %d, %Y') if pd.notna(x) else None)
            df['max_dependency_finish'] = df['max_dependency_finish'].apply(lambda x: x.strftime('%b %d, %Y') if pd.notna(x) else None)
            
            
            
            #Convert Duration to days
            df['Duration'] = pd.to_timedelta(df['Duration'], errors='coerce')
            df['Duration'] = df['Duration'].dt.days
            
            #Create Time to Finish
            df['Time to Finish'] = df['Finish_Date'] - now
            df['Time to Finish'] = df['Time to Finish'].dt.days
            
            #Create Time to Start
            df['Time to Start'] = df['Start_Date'] - now
            df['Time to Start'] = df['Time to Start'].dt.days
            
            conditions = [
                (df['Finish_Date'] <= now) & pd.isna(df['Actual_Finish']),
                 (df['Start_Date'] <= now) & pd.isna(df['Actual_Start']),
                (df['Finish_Date'] >= now) & (df['Actual_Start'] <= now),
                ~pd.isna(df['Actual_Finish']),
                (df['Start_Date'] >= now) & (pd.isna(df['Actual_Start']))
            ]
            values = ['Should Finished', 'Should Started','In Progress', 'Completed', 'Future Tasks']
            
            # use np.select() to create the 'Status' column
            df['Status'] = np.select(conditions, values, default='Delayed')
            
            #convert dt to days
            df['Total_Slack_Days'] = pd.to_timedelta(df['Total_Slack'], errors='coerce').dt.days
            
            #Create/Calculate Total Predecessors
            df['Total Predecessors'] = df['Predecessors'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else None)
            
            #state.df = df    
            return df






# Define the session state
state = st.session_state
# Check if an uploaded file exists in the session state
if 'uploaded_file' not in state:
    state.uploaded_file = None

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    state.uploaded_file = uploaded_file
    




#@st.cache_data
def run_pyg(df):
    # Generate the HTML using Pygwalker
    pyg_html = pyg.walk(df, return_html=True)  
    # Embed the HTML into the Streamlit app
    components.html(pyg_html, height=1000, scrolling=True)


    
try:
    df = process_data(uploaded_file)
    run_pyg(df)
except:
    st.warning("Alarm")



#if 'df' in state:
#    run_pyg(state.df)
        

    
    #df = pd.read_excel(uploaded_file, engine="openpyxl")
    #df = process_data(uploaded_file)
    
    columns_check = ['ID',
     'Active',
     'Task_Mode',
     'Task_Name',
     'Duration',
     'Start_Date',
     'Finish_Date',
     'Predecessors',
     'Resource_Names',
     'Actual_Start',
     'Actual_Finish',
     'Total_Slack',
     'Successors',
     'Unique_ID',
     'Outline_Level',
     'Constraint_Date',
     'Constraint_Type',
     'Early_Start',
     'Early_Finish',
     'Late_Start',
     'Late_Finish',
     'Notes']
    
#    try:
#        if df.columns.tolist() != columns_check:
#            raise ValueError("DataFrame columns do not match expected columns")
#    except ValueError as e:
#        st.error("Columns not match template!!")
#        st.warning("Check your dataframe: " + str(e))
#    else:
#        st.balloons()
#        st.success("Excel File was read Successfully!")
#        st.toast("Data begins to process, just a second...")
        
        # Generate the HTML using Pygwalker
        #pyg_html = pyg.walk(df, return_html=True)  
        # Embed the HTML into the Streamlit app
        #components.html(pyg_html, height=1000, scrolling=True)
        #if st.button("See Dataframe"):
        #    st.write(df)    
   
    #state.df = df

#@st.cache_data
#def run_pyg(df):
#    # Generate the HTML using Pygwalker
#    pyg_html = pyg.walk(df, return_html=True)  
#    # Embed the HTML into the Streamlit app
#    components.html(pyg_html, height=1000, scrolling=True)



#if 'df' in state:
#    run_pyg(state.df)





