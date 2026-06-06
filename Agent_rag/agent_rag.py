# from dotenv import load_dotenv
# load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver 
import streamlit as stream
import os

api_key = stream.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# data in the st session state

if "document_uploaded" not in stream.session_state:
    stream.session_state.document_uploaded = False

if "agent" not in stream.session_state:
    stream.session_state.agent = None

if "vector_store" not in stream.session_state:
    stream.session_state.vectorstore = None

if "messages" not in stream.session_state:
    stream.session_state.messages = []


def processing_documets(path):
# Load the PDF document
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create a vector store and add the document chunks

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = InMemoryVectorStore.from_documents(
        documents=texts,
        embedding=embeddings
    )

    #create an agent

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    @tool
    def retrieve_tool(query: str):
        """ A tool for retrieving information from the vector store. Always use this tool to get the relevant information from the vector store. Do not make up any information. If you don't know the answer, say you don't know.
        """
        context = ""

        docs = vectorstore.similarity_search(query, k=3)
        for doc in docs:
            context += doc.page_content + "\n" 

        return context






    system_prompt = """
        You are a document question-answering assistant.

        The user has uploaded one or more documents. Your job is to answer questions using only the information found in those uploaded documents.

        Rules:
        1. Always use the retrieve_tool before answering.
        2. Answer only from the retrieved document content.
        3. Do not use outside knowledge.
        4. Do not guess or make up information.
        5. If the answer is not found in the uploaded document, say: I don't know based on the uploaded document.
        6. If the question is not related to the uploaded document, say: I can only answer questions related to the uploaded document.
        7. Keep answers concise and direct.
        8. Do not include tool names, metadata, JSON, signatures, citations, or internal reasoning.
        9. Return only the final answer text.

        Answer style:
        - For simple factual questions, answer in one short sentence or phrase.
        - For list-based questions, use short bullet points.
        - For summary questions, provide a brief summary.
    """

    memory = InMemorySaver()   

    agent = create_agent(
        tools=[retrieve_tool], 
        model=llm, 
        system_prompt=system_prompt,
        checkpointer=memory
    )

    stream.session_state.agent = agent
    stream.session_state.document_uploaded = True

#upload UI

stream.subheader("AI Bot with RAG Tool 🤖")

if not stream.session_state.document_uploaded:
    uploaded_file = stream.file_uploader(label="Upload a PDF document", type=["pdf"], accept_multiple_files=True)

    if uploaded_file:
        with stream.spinner("Processing document..."):
            try:
                path = "/doc_files/"
                os.makedirs(path, exist_ok=True)

                for file in uploaded_file:
                    file_path = os.path.join(path, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())

                processing_documets(path)
                stream.rerun()
            except Exception as e:
                stream.error("Something went wrong while uploading or processing the document.")

#chat UI
if stream.session_state.document_uploaded and stream.session_state.agent:

    for message in stream.session_state.messages:
        role = message["role"]
        content = message["content"]
        stream.chat_message(role).markdown(content)

    query = stream.chat_input("Ask anything about the uploaded document 👨: ")


    if query:

        stream.session_state.messages.append({"role": "user", "content": query})

        stream.chat_message("user").markdown(query)
        try:
            response = stream.session_state.agent.invoke(
                {
                    "messages" : [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                },
                {
                    "configurable" : {
                        "thread_id": "1"
                    }
                }
            )
            # answer = response["messages"][-1].content
            content = response["messages"][-1].content
            if isinstance(content, list):
                answer = content[0]["text"]
            else:
                answer = content
            stream.chat_message("ai").markdown(answer)
            stream.session_state.messages.append({"role": "ai", "content": answer})
        except Exception as e:
            answer = "Something went wrong. Please try again."
            stream.chat_message("ai").markdown(answer)
            stream.session_state.messages.append({"role": "ai", "content": answer})
            print(e)


            

                    



# query = "What is the patient name?, and what is his age and, what is the name of the doctors, any vitamin deficency and movie name"

# response = agent.invoke({
#     "messages" : [
#         {
#             "role": "user",
#             "content": query
#         }
#     ]
# },
#         {
#             "configurable" : {
#                 "thread_id": "123"
#             }
#         })
# print(response["messages"][-1].content[0]["text"])