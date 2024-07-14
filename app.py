import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
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
    """
    Prompt template for QA retrieval for each vectorstore
    """
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


def create_vector_db():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls = PyPDFLoader)
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

@cl.on_chat_start
async def start():
    if not os.path.exists(DB_FAISS_PATH):
        msg = cl.Message(content="Creating vector database. This may take a while...")
        await msg.send()
        create_vector_db()
        await msg.update(content= "Vector database created. Starting the bot...")
    else:
        msg = cl.Message(content="Starting the bot...")
        await msg.send()

    chain = qa_bot()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)


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