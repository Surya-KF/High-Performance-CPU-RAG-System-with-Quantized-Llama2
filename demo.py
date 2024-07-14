import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl

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
                                           retriever=db.as_retriever(search_kwargs={'k': 3}),
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


def create_vector_db(pdf_file):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the PDF QA Bot. Please upload a PDF file.").send()

    pdf_file = None
    while pdf_file is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file",
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()

        if files:
            pdf_file = files[0]
        else:
            await cl.Message(content="No file was uploaded. Please try again.").send()

    msg = cl.Message(content="Processing your PDF... This may take a moment.")
    await msg.send()

    db = create_vector_db(pdf_file)

    chain = qa_bot()
    cl.user_session.set("chain", chain)

    msg.content = "PDF processing complete. You can now ask questions about your document!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()