# pip install -r requirements.txt

# python --version
# Python 3.8.18

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import time

# re-make this to store as array of dicts,
# to know sources where the text was from
# MODELS_DB_PAIRS = [
#     ('all-minilm:latest', '45 MB', 'minilm'),
#     ('gte-base:latest', '117 MB', 'gte'),
#     ('nomic-embed-text:latest', '274 MB', 'nomic'),
#     ('qwen2:0.5b', '352 MB', 'qwen'),
#     ('tinyllama:latest', '637 MB', 'tinyllama'),
#     ('mxbai-embed-large:latest', '669 MB', 'mxbai'),
#     ('gemma:2b-instruct-v1.1-q2_K', '1.2 GB', 'gemma'),
#     ('phi3:mini-128k', '2.2 GB', 'phimini'),
#     ('llama3:latest', '4.7 GB', 'llama'),
# ]

load_dotenv() # gives access to langchain access to api-keys in .env
DB_NAME = 'minilm'
EMBEDDING_MODEL_NAME = 'all-minilm:latest'
CONNECTION = f"postgresql+psycopg://postgres:password@localhost:5432/{DB_NAME}"
# LLM = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.5, "max_length": 512})
LLM = Ollama(model='llama3:latest', base_url='http://localhost:31415')

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore():
    # takes approx 12s for 71.MB file
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url='http://localhost:31415')
    vectorstore = PGVector(
            embeddings=embedding,
            collection_name='mind',
            connection=CONNECTION,
            use_jsonb=True,
            )
    return vectorstore


def initialize_empty_conversation():
    # Initialize a basic LLM and retriever with empty or placeholder components
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create an empty vectorstore (FAISS) or a mock retriever
    # For demonstration, we're assuming an empty FAISS vectorstore can be created
    vectorstore = get_vectorstore()
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_conversation_chain(vectorstore):

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    prompt = f"Reference the given documents that answer the following question: {user_question}"
    our_response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = our_response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = initialize_empty_conversation()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                time_message = st.empty()  # Placeholder for time display
                with st.spinner("Processing... Refresh page to stop"):
                    start_time = time.time()  # Start time tracker
                    try:
                        vectorstore = get_vectorstore()
                        st.session_state.conversation = get_conversation_chain(vectorstore)

                        current_time = time.time() - start_time
                        time_message.text(f"Elapsed time: {current_time:.2f} seconds")

                        st.success("Processing completed successfully!")  # Display success message

                        

                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
            else:
                st.warning("Please upload at least one PDF.")
    #st.session_state.conversation #makes the conversation available outside the scope, and also to not re


if __name__ == '__main__':
    main()
