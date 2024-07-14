import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved at {DB_FAISS_PATH}")

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        if "allow_dangerous_deserialization" in str(e):
            print("Warning: Allowing dangerous deserialization. Ensure you trust the source of your FAISS index.")
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            raise e

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Streamlit UI
st.title("Medical Bot")

if not os.path.exists(DB_FAISS_PATH):
    st.write("Creating vector database. This may take a while...")
    create_vector_db()
    st.write("Vector database created. You can now start the bot.")

query = st.text_input("Enter your query:")
if query:
    response = final_result(query)
    st.write("Response:", response['result'])
    st.write("Sources:", response['source_documents'])
