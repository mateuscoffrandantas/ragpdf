import os
import tempfile
import pandas as pd
import pytesseract
from PIL import Image
import docx
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
from pypdf.errors import PdfReadError
from openai import OpenAIError

# Add app name
st.subheader("Q&A with AI - NLP using LangChain")

# Interactive components
file_input = st.file_uploader("Upload a file", type=['pdf', 'txt', 'csv', 'docx', 'jpeg', 'png'])
prompt = st.text_area("Enter your questions", height=160)
run_button = st.button("Run!")

select_k = st.slider("Number of relevant chunks", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Chain type", ['stuff', 'map_reduce', "refine", "map_rerank"])

# Define model and API key
openai_key = st.text_input("Enter your OpenAI API Key", type='password')

# Function to load documents
def load_document(file_path, file_type):
    try:
        if file_type == 'application/pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_type == 'text/plain':
            loader = TextLoader(file_path)
            return loader.load()
        elif file_type == 'text/csv':
            df = pd.read_csv(file_path)
            return [{"page_content": df.to_string()}]
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = docx.Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            return [{"page_content": "\n".join(full_text)}]
        elif file_type in ['image/jpeg', 'image/png']:
            text = pytesseract.image_to_string(Image.open(file_path))
            return [{"page_content": text}]
        else:
            st.error("Unsupported file type.")
            return None
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

# Q&A function
def qa(file_path, file_type, query, chain_type, k):
    try:
        documents = load_document(file_path, file_type)
        if not documents:
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            st.error("No text content found in the document.")
            return None
        
        # Configure OpenAI if key provided
        if not openai_key:
            st.error("OpenAI API Key is required!")
            return None
            
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI: {str(e)}")
            return None
            
        # Create vectorstore using FAISS instead of Chroma
        try:
            db = FAISS.from_documents(texts, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa({"query": query})
        return result

    except PdfReadError as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None
    except OpenAIError as e:
        st.error(f"API Key error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

# Function to display result in Streamlit
def display_result(result):
    if result:
        st.markdown("### Result:")
        st.write(result["result"])
        st.markdown("### Relevant source text:")
        for doc in result["source_documents"]:
            st.markdown("---")
            st.markdown(doc.page_content)

# App execution
if run_button and file_input and prompt and openai_key:
    with st.spinner("Running QA..."):
        try:
            # Save file to temporary location
            temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_input.read())

            # Run Q&A function
            result = qa(temp_file_path, file_input.type, prompt, select_chain_type, select_k)

            # Display result
            display_result(result)
            
            # Clean up temporary file
            try:
                os.remove(temp_file_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")
