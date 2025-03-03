# import necessary libraries
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore 
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
import tempfile
import sys
import os
import base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pdf_project.llm import talk

# import llm and its prompt from package
llm, prompt = talk.export_llm()

# create chunk from the documents
def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len,
        separators=["\n\n", ".", "!", "?"],
        )
    chunks = text_splitter.split_documents(docs)
    return chunks

# load the pdf using PyMuPDFLoader
def process_pdf(pdf_file):
    """Processes the uploaded PDF using PyMuPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
    #loader = UnstructuredPDFLoader(temp_pdf_path)
    loader = PyMuPDFLoader(temp_pdf_path)
    docs = loader.load()
    return docs

# create embeddings and store it in vectorstore & bm25
def embed_docs(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=chunks)
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=word_tokenize)
    return vector_store, bm25_retriever

def create_vectorstore():
    pass

# make the context more readable
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

# question answering llm
def chat_with_llm(question):
    for output in qna_chain.stream(question):
        yield output

# display the preview of pdf 
def display_pdf(uploaded_file):
    """Convert a PDF file to a base64 string and display it in an iframe."""
    base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="320" height="400" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# streamlit interface
st.title("Talk with PDF")
st.write("Upload your PDF")

if "pdf_file" not in st.session_state:
    st.session_state["pdf_file"] = None
if "report_clicked" not in st.session_state:
    st.session_state["report_clicked"] = False
if "summ_clicked" not in st.session_state:
    st.session_state["summ_clicked"] = False
if "talk_clicked" not in st.session_state:
    st.session_state["talk_clicked"] = False

# upload file
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf", accept_multiple_files=False)

container_pdf, container_talk = st.columns([2, 2])

with container_pdf:
    # ensure the file is uploaded
    if uploaded_file is not None:
        if "uploaded_filename" not in st.session_state or st.session_state["uploaded_filename"] != uploaded_file.name:
            st.session_state["uploaded_filename"] = uploaded_file.name
            with st.spinner(text="Processing PDF.."):
                pdf_text = process_pdf(uploaded_file)
                chunk = chunk_docs(pdf_text)
                vector_store, bm25_retriever = embed_docs(chunk)
            st.session_state["vector_store"] = vector_store
            st.session_state["bm25_retriever"] = bm25_retriever
            st.session_state["pdf_file"] = uploaded_file
            st.session_state["docs_text"] = pdf_text
            st.write(st.session_state["uploaded_filename"])
            st.session_state["report_clicked"] = None
            st.session_state["talk_clicked"] = None

        else:
            vector_store = st.session_state["vector_store"]
            bm25_retriever = st.session_state["bm25_retriever"]
        
    else:
        vector_store=None
        bm25_retriever=None
    # show the pdf is file is successfully uploaded
    if st.session_state["pdf_file"]:
        display_pdf(uploaded_file)

# ensure that it won't loop from creating embedding
if vector_store:
    vector_store = st.session_state["vector_store"]
    vector_retriever = vector_store.as_retriever(search_type = 'mmr', 
                                     search_kwargs = {'k': 3, 'fetch_k': 20, 'lambda_mult': 1})
    vector_retriever = vector_store.as_retriever()
    bm25_retriever = st.session_state["bm25_retriever"]
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

    with container_talk:
        col1, col2 = st.columns(2)
        # button logic
        with col1:
            if st.button('Generate report'):
                st.session_state["report_clicked"] = True
                st.session_state["talk_clicked"] = None

        with col2:
            if st.button('Talk'):
                st.session_state["talk_clicked"] = True
                st.session_state["report_clicked"] = None
            
        if st.session_state["report_clicked"]:
            with st.spinner("Generating report.."):
                context = format_docs(hybrid_retriever.invoke("generate report"))
                report_chain = prompt | llm | StrOutputParser()
                response = st.write_stream(response for response in report_chain.stream({'context': context, 'question': "Generate a comprehensive and well-structured report based on the provided context. Ensure clarity, depth, and coherence in your analysis. Format the response in Markdown for readability and proper structuring."}))

        if st.session_state["talk_clicked"]:
            user_input = st.text_input("Your question: ")
            qna_chain = (
            {"context": hybrid_retriever|format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            #st.text_area("context:", format_docs(hybrid_retriever.invoke(user_input)))
            if user_input:
                st.write("Answer: ")
                st.write_stream(chat_with_llm(user_input))
                #st.write(user_input)




