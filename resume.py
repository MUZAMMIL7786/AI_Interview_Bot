import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, LLMChain, RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory



from dotenv import load_dotenv

load_dotenv()

os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

st.title("Chatgroq With Llama3 Demo")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

prompt=ChatPromptTemplate.from_template(
"""

You are an expert Data Scientist, provide with a task of taking interview of a candidate for the role of Data Scientist
with 1 year of experience.
Take account of resume and try to intersect the areas that can be asked for the interviews
Try to keep complexity as minimal as possible , Ask one question at a time and based on the answer eceived then
build another question on top of that.
Do not generate questions which is out of context with respect to the role.
Do not answer the question
Try to build questions which intersect both resume and job role.
If candidate types anything that dosent answer the question tell him to be serious with the inerview

Dont use okay, lets start the interview multiple times

Start questions when candidate tell "Start the Inteerview"
<context>
{context}
<context>
Ansswer:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFLoader("MOHAMMED_Resume_2022_Grad.pdf") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


def combined_chain(query):
    # Retrieve relevant documents
    docs = qa_chain.run(query)

    # Process documents and generate response
    response = llm_chain.run(docs)

    # Update conversation buffer
    memory.add_user_message(query)
    memory.add_bot_message(response)

    return response


prompt1=st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time
memory = ConversationBufferMemory()


if prompt1:
    retriever = st.session_state.vectors.as_retriever()
    qa_chain = RetrievalQA.from_llm(llm, retriever=retriever)

    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    response = combined_chain("start the interview")
    st.write(response['answer'])





