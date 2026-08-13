import streamlit as st
import os
import base64
import time
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from pinecone import Pinecone

# --- Page Configuration ---
st.set_page_config(page_title="VX/IP-Series Hybrid Telecom Assistant", layout="centered")

# --- USER DATABASE ---
USER_DB = {
    "Dudub": "dudu1408,technician",
    "Customer": "cust1234,customer"
}

if "users" in st.secrets:
    USER_DB.update(st.secrets["users"])

ROLE_FILTERS = {
    "customer": {"role": "customer"},
    "technician": {"role": {"$in": ["customer", "technician"]}}
}

# --- Setup Keys ---
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

INDEX_NAME = "manuals-index"

# --- Authentication Logic ---
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

def login_ui():
    st.title("Secure Technical Portal")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
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

# --- Sidebar Controls & Model Selection ---
st.sidebar.success(f"Logged in as: {st.session_state['user_role'].capitalize()}")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Settings")

# 1. Chat Model Dropdown Selection
MODEL_OPTIONS = {
    "Gemini 2.5 Flash (Google - Multimodal & Fast)": "gemini-2.5-flash",
    "Qwen 2.5 72B Instruct (HF - Deep Telecom/Engineering)": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.3 70B Instruct (HF - Open Telecom Standards)": "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen 3.8": "Qwen/Qwen3.8-2.4T-A95B",
    "mistraAI": "mistralai/Mistral-7B-Instruct-v0.3"
}

selected_model_label = st.sidebar.selectbox("Select Reasoning / Chat Model:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_label]

# 2. Embedding Engine Selection
EMBEDDING_OPTIONS = {
    "BGE Base English (HuggingFace 768d - High Density)": "hf_bge",
    "Gemini Embeddings (Google 768d)": "gemini"
}
selected_embedding_type = st.sidebar.selectbox("Select Embedding Engine:", list(EMBEDDING_OPTIONS.keys()))

if st.sidebar.button("Logout"):
    st.session_state['user_role'] = None
    st.rerun()

# --- Model Initialization Factories ---
@st.cache_resource
def get_embedding_engine(engine_type):
    """Loads 768-dimension embeddings for Pinecone vector space compatibility."""
    if engine_type == "hf_bge":
        return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    else:
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=768)

@st.cache_resource
def get_vision_llm():
    """Dedicated Multimodal Vision Engine for PDF Processing."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

def get_chat_llm(model_id):
    """Instantiates the user-selected Chat Reasoning model."""
    if "gemini" in model_id:
        return ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    else:
        hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))
        if not hf_token:
            st.error("⚠️ Hugging Face API Token missing in st.secrets! Defaulting back to Gemini.")
            return get_chat_llm("gemini-2.5-flash")
        
        endpoint = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=1024,
            do_sample=False,
            huggingfacehub_api_token=hf_token
        )
        return ChatHuggingFace(llm=endpoint)

# Initialize Active Components
embeddings = get_embedding_engine(EMBEDDING_OPTIONS[selected_embedding_type])
vision_llm = get_vision_llm()
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# --- Active Manuals Sidebar Status ---
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Active Manuals in DB")

try:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    idx = pc.Index(INDEX_NAME)
    stats = idx.describe_index_stats()
    total_count = stats.get('total_vector_count', 0)
    st.sidebar.caption(f"Total Vector Chunks: {total_count}")
    
    if total_count > 0:
        response = idx.query(vector=[0.1] * 768, top_k=1000, include_metadata=True)
        unique_files = {match["metadata"]["source"] for match in response.get("matches", []) if "metadata" in match and "source" in match["metadata"]}
        for file in unique_files:
            st.sidebar.markdown(f"- 📄 `{file}`")
    else:
        st.sidebar.warning("Database is currently empty.")
except Exception:
    st.sidebar.caption("Connect an index to view library status.")

# --- Admin Function: Multimodal Resumable PDF Ingestion ---
if st.session_state['user_role'] == 'technician':
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("Admin: Upload Manual", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        doc_role = st.selectbox("Assign Access Level", ["customer", "technician"])
        
        if uploaded_file and st.button("Process & Secure"):
            status_box = st.status("Initializing Heavy-Duty Processing...", expanded=True)
            
            try:
                # 1. Checkpoint verification
                last_processed_page = 0
                try:
                    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                    idx = pc.Index(INDEX_NAME)
                    response = idx.query(
                        vector=[0.0] * 768, 
                        filter={"source": uploaded_file.name}, 
                        top_k=10000, 
                        include_metadata=True
                    )
                    pages_found = [match['metadata'].get('page', 0) for match in response.get('matches', []) if 'metadata' in match and 'page' in match['metadata']]
                    if pages_found:
                        last_processed_page = int(max(pages_found))
                except Exception as e:
                    status_box.write(f"⚠️ Checkpoint notice: Starting fresh. ({str(e)})")

                pdf_bytes = uploaded_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                total_pages = len(doc)
                
                if last_processed_page >= total_pages:
                    status_box.update(label="✅ Document already fully processed!", state="complete")
                    st.stop()
                
                status_box.update(label=f"PDF Loaded. Resuming from page {last_processed_page + 1} of {total_pages}...", state="running")
                progress_bar = st.progress(last_processed_page / total_pages)
                
                # 2. Page-by-page streaming ingestion
                BATCH_SIZE = 1 
                
                for i in range(last_processed_page, total_pages):
                    current_page_num = i + 1
                    page = doc.load_page(i)
                    
                    image_list = page.get_images()
                    needs_vision = False
                    
                    if image_list:
                        for img in image_list:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if base_image:
                                width = base_image.get("width", 0)
                                height = base_image.get("height", 0)
                                if width > 200 and height > 200:
                                    needs_vision = True
                                    break
                    
                    if needs_vision:
                        status_box.write(f"📸 Page {current_page_num}/{total_pages}: Large diagram detected. Processing with Gemini Vision...")
                        pix = page.get_pixmap(dpi=72)
                        img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                        
                        prompt = """
                        You are a technical documentation assistant. 
                        1. Extract all text from this manual page exactly as written.
                        2. If there are any screenshots, diagrams, tables, or UI panels, write a highly detailed description of them.
                        Format your response cleanly.
                        """
                        
                        message = HumanMessage(
                            content=[
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        )
                        
                        success = False
                        for attempt in range(3):
                            try:
                                response = vision_llm.invoke([message])
                                page_content = response.content
                                success = True
                                break
                            except Exception as api_error:
                                if "429" in str(api_error).lower() or "quota" in str(api_error).lower():
                                    status_box.write(f"⏱️ Rate limit hit. Pausing 10s...")
                                    time.sleep(10)
                                else:
                                    raise api_error
                        if not success:
                            raise Exception("Rate limit retry failed.")
                    else:
                        status_box.write(f"📄 Page {current_page_num}/{total_pages}: Direct text extraction...")
                        page_content = page.get_text()
                    
                    # Splitting and Immediate DB Injection
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents([Document(page_content=page_content, metadata={"page": current_page_num})])
                    
                    for chunk in chunks:
                        chunk.metadata.update({"role": doc_role, "source": uploaded_file.name})
                        
                    vectorstore.add_documents(chunks)
                    progress_bar.progress(current_page_num / total_pages)
                
                status_box.update(label=f"✅ Processing completed: {uploaded_file.name}", state="complete", expanded=False)
                st.success(f"Storage complete for {uploaded_file.name}!")
                
            except Exception as e:
                status_box.update(label=f"❌ Stopped at Page {current_page_num}", state="error")
                st.error(f"Pipeline Error: {str(e)}")
                st.info("Your progress up to the last saved page is securely stored in Pinecone. Click 'Process & Secure' again to resume.")

    # Emergency Wipe
    with st.sidebar.expander("Admin: Danger Zone", expanded=False):
        if st.button("⚠️ Wipe Entire Database"):
            try:
                pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
                index = pc.Index(INDEX_NAME)
                index.delete(delete_all=True)
                st.success("Database completely cleared!")
            except Exception as e:
                st.error(f"Failed to clear database: {str(e)}")

# --- Main Search & Telecom Chat Interface ---
st.title("🛠️ IP50 / VX-Series Telecom Assistant")
st.caption(f"Active Chat Reasoning Engine: **{selected_model_label}**")

query = st.text_input("Ask a technical/telecom engineering question:")

if query:
    with st.spinner(f"Retrieving vector context & generating response using {selected_model_id}..."):
        user_filter = ROLE_FILTERS[st.session_state['user_role']]
        retriever = vectorstore.as_retriever(search_kwargs={"filter": user_filter, "k": 4})
        docs = retriever.invoke(query)
        
        if not docs:
            st.warning("No relevant information found within authorized manuals.")
        else:
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"""You are an expert telecom and wireless hardware engineering assistant.
Answer the following query using ONLY the provided technical documentation context.
If the context does not contain enough detail, state what is missing clearly.

Context:
{context}

Question: {query}"""
            
            try:
                chat_llm = get_chat_llm(selected_model_id)
                response = chat_llm.invoke(prompt)
                
                st.subheader("Engineering Response")
                st.write(response.content if hasattr(response, 'content') else str(response))
            except Exception as e:
                st.error(f"Chat Model Generation Error ({selected_model_id}): {str(e)}")
            
            st.markdown("---")
            with st.expander("View Retrieved Source Context Chunks"):
                for doc in docs:
                    st.info(f"**Source:** {doc.metadata.get('source', 'Unknown')} | **Page:** {doc.metadata.get('page', 'N/A')} | **Access Level:** {doc.metadata.get('role', 'none').capitalize()}\n\n{doc.page_content[:300]}...")
