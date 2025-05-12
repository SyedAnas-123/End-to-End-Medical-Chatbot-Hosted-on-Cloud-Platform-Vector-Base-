from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPEN_AI_API_KEY =os.environ.get('OPEN_AI_API_KEY')
OPEN_AI_BASE_URL = os.environ.get('OPEN_AI_BASE_URL')


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPEN_AI_API_KEY"] = OPEN_AI_API_KEY
os.environ["OPEN_AI_BASE_URL"] = OPEN_AI_BASE_URL


embedings = download_hugging_face_embedings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding= embedings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    openai_api_base= OPEN_AI_BASE_URL,
    openai_api_key= OPEN_AI_API_KEY ,
    temperature=0.4,
    max_tokens=500
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # default 10000, or Render-defined port
    app.run(host="0.0.0.0", port=port)
