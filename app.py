import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone




# --- TEMPORARY HARDCODED DATABASE TO BYPASS CACHE ---
USER_DB = {
    "Dudub": "dudu1408,technician",
    "Customer": "cust1234,customer"
}

ROLE_FILTERS = {
    "customer": {"role": "customer"},
    "technician": {"role": {"$in": ["customer", "technician"]}}
}

# --- Setup Keys ---
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
INDEX_NAME = "manuals-index"

# --- Authentication Logic ---
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

def login_ui():
    st.title("Secure Technical Portal")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Case-insensitive matching
        username_lower = username.lower()
        user_db_lower = {k.lower(): v for k, v in USER_DB.items()}
        
        if username_lower in user_db_lower:
            secret_string = user_db_lower[username_lower]
            saved_password, saved_role = secret_string.split(",")
            
            if password == saved_password.strip():
                st.session_state['user_role'] = saved_role.strip()
                st.rerun()
            else:
                st.error("Invalid password.")
        else:
            st.error("Username not found.")

if not st.session_state['user_role']:
    login_ui()
    st.stop()



st.sidebar.success(f"Logged in as: {st.session_state['user_role'].capitalize()}")
if st.sidebar.button("Logout"):
    st.session_state['user_role'] = None
    st.rerun()

# --- Initialize AI & Database ---
@st.cache_resource
def init_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return llm, vectorstore

llm, vectorstore = init_rag()

# --- Admin Function: Uploading Manuals ---
if st.session_state['user_role'] == 'technician':
    with st.sidebar.expander("Admin: Upload Manual"):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        doc_role = st.selectbox("Assign Access Level", ["customer", "technician"])
        
        if uploaded_file and st.button("Process & Secure"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Read, chunk, and tag the document
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)
            
            # Inject metadata tag for security

                        # Inject metadata tag for security
            for chunk in chunks:
                chunk.metadata = {"role": doc_role, "source": uploaded_file.name}
                
            # --- NEW TRY/EXCEPT BLOCK ---
            try:
                vectorstore.add_documents(chunks)
                st.success("Manual secured and loaded into Vector DB!")
            except Exception as e:
                st.error(f"Google API Error: {str(e)}")

            

# --- Main Search Interface ---
st.title("🛠️ VX-Series Assistant")
query = st.text_input("Ask a technical question:")

if query:
    with st.spinner("Searching authorized documents..."):
        # 1. Enforce Security: Get the filter for the logged-in user
        user_filter = ROLE_FILTERS[st.session_state['user_role']]
        
        # 2. Retrieve ONLY authorized chunks
        retriever = vectorstore.as_retriever(search_kwargs={"filter": user_filter, "k": 4})
        docs = retriever.invoke(query)
        
        if not docs:
            st.warning("No relevant information found within your authorized manuals.")
        else:
            # 3. Formulate the answer
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"Answer the question using ONLY the context provided.\nContext: {context}\nQuestion: {query}"
            
            response = llm.invoke(prompt)
            st.write(response.content)
            
            with st.expander("View Source Citations"):
                for doc in docs:
                    st.info(f"Source: {doc.metadata['source']} | Role: {doc.metadata['role']}\n\n{doc.page_content[:200]}...")
