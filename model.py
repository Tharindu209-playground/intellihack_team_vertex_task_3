from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"


custom_prompt_template = """

You are a customer service representative loan section at a bank. A customer has asked you a question.
You need to provide an answer to the customer.

Context: {context}
Question: {question}

Above is the context and question. As for the situation you may or may not use above context to answer the question.
You can also use your knowledge to answer the question.
But make sure the answer is relevant to the question asked by the customer.

mainly you need to provide answers for the following domains:
1. Loan Eligibility Check
2. Loan Products Information
3. Loan Application Process Guidance
4. FAQs and Troubleshooting
5. Personalized Recommendations

"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )
    return prompt


def return_retriever(db):
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return retriever


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=return_retriever(db),
        return_source_documents=False,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                input_key="question"
            ),
        },
    )
    
    return qa_chain


# Loading the model
def load_llm():
    # Load the chatGPT model
    llm = ChatOpenAI(model='gpt-3.5-turbo')
    return llm


# QA Bot
def chat_bot():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = load_llm()
    prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, prompt, db)

    return qa_chain


# output function
def final_result(query):
    result = chat_bot()
    response = result({"query": query})
    return response


# chainlit code
@cl.on_chat_start
async def start():
    chain = chat_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Smart Bank. How can I help you today?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(
        message.content, callbacks=[cb]
    ) 

    answer = res["result"]

    await cl.Message(content=answer).send()
