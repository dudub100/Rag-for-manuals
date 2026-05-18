import streamlit as st
import os
import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- Page Configuration ---
st.set_page_config(page_title="VX-Series Technical Assistant", layout="centered")

# --- USER MANAGEMENT & DATABASE MERGING ---
# 1. Start with your temporary hardcoded database
USER_DB = {
    "Dudub": "dudu1408,technician",
    "Customer": "cust1234,customer"
}

# 2. Check if a [users] section exists in Streamlit Secrets, and merge it
if "users" in st.secrets:
    secrets_users = st.secrets["users"]
    USER_DB.update(secrets_users)

# --- DEBUG PRINT (Outputs to Terminal/Streamlit Cloud Logs) ---
print("\n=== DEBUG: FINAL USER DATABASE ===")
for username, data in USER_DB.items():
    print(f"Loaded User -> Username: '{username}' | Data: '{data}'")
print("===================================\n")

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
    
    # Optional: Uncomment the line below if you want to see the database on screen during testing
    # st.write("Debug - System is currently seeing these users:", USER_DB)
    
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

# --- Sidebar Elements ---
st.sidebar.success(f"Logged in as: {st.session_state['user_role'].capitalize()}")
if st.sidebar.button("Logout"):
    st.session_state['user_role'] = None
    st.rerun()

# --- Initialize AI & Database ---
@st.cache_resource
def init_rag():
    # Current active embedding model (Compressing output to match 768 dimensions)
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

# --- Admin Function: Uploading & Managing Manuals (Multimodal Vision Engine) ---
if st.session_state['user_role'] == 'technician':
    st.sidebar.markdown("---")
    
    # Section 1: Multimodal PDF Upload
    with st.sidebar.expander("Admin: Upload Manual", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        doc_role = st.selectbox("Assign Access Level", ["customer", "technician"])
        
        if uploaded_file and st.button("Process & Secure"):
            import fitz  # PyMuPDF
            from langchain_core.messages import HumanMessage
            from langchain_core.documents import Document
            
            with st.spinner("Analyzing pages and UI screenshots with Gemini Vision..."):
                # Read PDF bytes directly from memory
                pdf_bytes = uploaded_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                enriched_pages = []
                progress_bar = st.progress(0)
                
                # Iterate through pages, convert to high-res images, and process via LLM
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=150)  # Render page image
                    img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                    
                    # Structuring the contextual vision instructions
                    prompt = """
                    You are a technical documentation assistant. 
                    1. Extract all text from this manual page exactly as written.
                    2. If there are any screenshots, diagrams, tables, or UI panels, write a highly detailed description of them. Include specific button names, field targets, IP addresses, toggles, or exact data values visible in the image.
                    Format your response cleanly.
                    """
                    
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    )
                    
                    # Invoke Gemini to read and inspect the page artifact
                    response = llm.invoke([message])
                    enriched_pages.append(Document(page_content=response.content, metadata={"page": i+1}))
                    
                    # Advance the layout progress indicator
                    progress_bar.progress((i + 1) / len(doc))
                
                # Chunk the combined text and vision descriptions
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(enriched_pages)
                
                # Inject security permissions and filename tracking metadata
                for chunk in chunks:
                    chunk.metadata.update({"role": doc_role, "source": uploaded_file.name})
                    
                # Store structural embeddings into Pinecone
                try:
                    vectorstore.add_documents(chunks)
                    st.success(f"Successfully loaded and vision-processed: {uploaded_file.name}")
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
            
            # Request answer translation and engineering formulation from LLM
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
