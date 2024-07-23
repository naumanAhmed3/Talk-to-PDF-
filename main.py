# -*- coding: utf-8 -*-
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import traceback
import os

# api_key = "AIzaSyAW9j_-ZOMQKo2HZx9YjQmk0hlu8k6e6-w"
api_key = os.getenv("GEMINI_API_KEY") # Add env variable pointing to your gemini key

# Initialize conversation history
conversation_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(f"Text Chunks: {len(chunks)}")  # Debugging statement
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store created and saved locally.")  # Debugging statement

def get_conversational_chain():
    prompt_template = """
    Maintain the flow of conversation based on the previous context and provide detailed answers. If the answer is not available in the context, say "answer is not available in the context".
    
    Conversation History: 
    {history}
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    model = GoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def handle_response(response):
    try:
        # Process the response and handle finish_reason correctly
        if 'output_text' in response:
            return response['output_text']
        else:
            return "Response does not contain 'output_text'"
    except AttributeError as e:
        st.error(f"Attribute Error: {str(e)}")
        st.error(traceback.format_exc())
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error(traceback.format_exc())
        return None

def user_input(user_question):
    global conversation_history

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Debugging statements
        print(f"Found {len(docs)} documents for the query.")
        for doc in docs:
            print(doc.page_content[:200])  # Print the first 200 characters of each document

        # Update conversation history
        conversation_history.append(f"User: {user_question}")

        chain = get_conversational_chain()
        response = chain.invoke(
            {"input_documents": docs, "history": "\n".join(conversation_history), "question": user_question},
            return_only_outputs=True
        )

        print("Response:", response)  # Debugging statement
        st.write("Response:", response)  # Debugging statement for Streamlit UI

        output_text = handle_response(response)
        if output_text:
            conversation_history.append(f"AI: {output_text}")
            st.write("Reply: ", output_text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(traceback.format_exc())

import streamlit as st

def main():
    st.set_page_config(page_title="Chat PDF")

    # Set background color
    st.markdown(
        """
        <style>
        .main {
            background-color: #ff836f;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Write keywords for what you want to look for in PDF")

    user_question = st.text_input("Ask from the knowledge base")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                print(f"Extracted Text Length: {len(raw_text)}")  # Debugging statement
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

