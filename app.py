import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- Page Configuration ---
st.set_page_config(page_title="VX-Series Technical Assistant", layout="centered")

# --- TEMPORARY HARDCODED DATABASE TO BYPASS CACHE ---

# 1. Start with your temporary hardcoded database
USER_DB = {
    "Dudub": "dudu1408,technician",
    "Customer": "cust1234,customer"
}

# 2. Check if a [users] section exists in Streamlit Secrets, and merge it
if "users" in st.secrets:
    # This safely pulls everything under the [users] section from secrets
    secrets_users = st.secrets["users"]
    
    # Merge the secrets dictionary into our hardcoded dictionary
    # (If a username exists in both, the secrets file value will override the hardcoded one)
    USER_DB.update(secrets_users)

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
    
    # Temporary UI debug line - delete this after you finish testing!
    st.write("Debug - System is currently seeing these users:", USER_DB)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    # ... rest of your login logic stays the same 
        
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

# --- Sidebar Elements ---
st.sidebar.success(f"Logged in as: {st.session_state['user_role'].capitalize()}")
if st.sidebar.button("Logout"):
    st.session_state['user_role'] = None
    st.rerun()

# --- Initialize AI & Database ---
@st.cache_resource
def init_rag():
    # Current active embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        output_dimensionality=768
    )
    
    # Active generation model with technical safety adjustments
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return llm, vectorstore

llm, vectorstore = init_rag()

# --- Admin Function: Uploading & Managing Manuals ---
if st.session_state['user_role'] == 'technician':
    st.sidebar.markdown("---")
    
    # Section 1: Upload Manuals
    with st.sidebar.expander("Admin: Upload Manual", expanded=False):
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
            for chunk in chunks:
                chunk.metadata = {"role": doc_role, "source": uploaded_file.name}
                
            # Process upload to Pinecone
            try:
                vectorstore.add_documents(chunks)
                st.success(f"Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Google API Error: {str(e)}")

    # Section 2: Emergency Database Management
    with st.sidebar.expander("Admin: Danger Zone", expanded=False):
        st.warning("Wiping the index will remove all uploaded vectors and context permanently.")
        if st.button("⚠️ Wipe Entire Database"):
            try:
                pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                index = pc.Index(INDEX_NAME)
                index.delete(delete_all=True)
                st.success("Database completely cleared!")
            except Exception as e:
                st.error(f"Failed to clear database: {str(e)}")

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
            # Formulate the answer
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"Answer the question using ONLY the context provided.\nContext: {context}\nQuestion: {query}"
            
            # Request translation and structure from LLM
            try:
                response = llm.invoke(prompt)
                st.subheader("System Response")
                st.write(response.content)
            except Exception as e:
                st.error(f"Chat Model Error: {str(e)}")
            
            # Display source files for compliance auditing
            st.markdown("---")
            with st.expander("View Source Citations"):
                for doc in docs:
                    st.info(f"**Source:** {doc.metadata.get('source', 'Unknown')} | **Access Level:** {doc.metadata.get('role', 'none').capitalize()}\n\n{doc.page_content[:250]}...")
