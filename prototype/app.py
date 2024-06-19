#pip install -r requirements.txt

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS #faiss runs locally --> use cloud for persistent storage
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
#from langchain.llms import HuggingFacePipeline

#re-make this to store as array of dicts, to know sources where the text was from
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# def get_conversation_chain(vectorstore):
#     model_name = "google/flan-t5-xxl"
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
#     llm = HuggingFacePipeline(pipeline=pipe)
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def handle_userinput(user_question):
#     our_response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = our_response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             #
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def handle_userinput(user_question):
    if st.session_state.conversation:
        our_response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = our_response['chat_history']

        #st.write(st.session_state.chat_history)

        for i, message in enumerate(st.session_state.chat_history):
            #st.write(message)
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Please process the PDFs first to initialize the conversation.")



def main():
    load_dotenv() #gives access to langchain access to api-keys in .env

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    #st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True) #allow unsafe html as it otherwise wouldn't process the html
    #st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                st.session_state.conversation = get_conversation_chain(vectorstore)

    st.session_state.conversation #makes the conversation available outside the scope, and also to not re


if __name__ == '__main__':
    main()