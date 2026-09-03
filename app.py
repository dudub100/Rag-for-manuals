import streamlit as st
import os
import base64
import time
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage

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



# 1. Chat Model Dropdown Selection (Universal Router)
MODEL_OPTIONS = {
    "Gemini 2.5 Flash (Google)": "gemini:gemini-2.5-flash",
    "GPT-OSS 120B (Groq - Deep Logic & Telecom, Free)": "groq:openai/gpt-oss-120b",
    "GPT-OSS 20B (Groq - Lightning Fast, Free)": "groq:openai/gpt-oss-20b",
    "Auto-Free Open Source (OpenRouter)": "openrouter:openrouter/free",
    "Llama 3 8B (OpenRouter - Free)": "openrouter:meta-llama/llama-3.1-8b-instruct:free"
}

selected_model_label = st.sidebar.selectbox("Select Reasoning / Chat Model:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[selected_model_label]

# 2. Embedding Engine Selection
EMBEDDING_OPTIONS = {
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



def get_chat_llm(selected_value):
    """Instantiates the user-selected Chat Reasoning model across multiple providers."""
    # Split the string to find the provider and the actual model ID
    # e.g. "groq:llama-3.3-70b-versatile" -> provider="groq", model_id="llama-3.3-70b-versatile"
    provider, model_id = selected_value.split(":", 1)
    
    if provider == "gemini":
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
        
    elif provider == "groq":
        groq_key = st.secrets.get("GROQ_API_KEY")
        if not groq_key:
            st.error("⚠️ GROQ_API_KEY missing in secrets! Defaulting to Gemini.")
            return get_chat_llm("gemini:gemini-2.5-flash")
            
        return ChatGroq(
            groq_api_key=groq_key,
            model_name=model_id,
            temperature=0.01,
            max_tokens=1024
        )
        
    elif provider == "openrouter":
        or_key = st.secrets.get("OPENROUTER_API_KEY")
        if not or_key:
            st.error("⚠️ OPENROUTER_API_KEY missing in secrets! Defaulting to Gemini.")
            return get_chat_llm("gemini:gemini-2.5-flash")
            
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
            model=model_id,
            temperature=0.01,
            max_tokens=1024
        )


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


# --- 4. Define Agent Tools ---

@tool
import math
import numpy as np
from scipy.optimize import root_scalar
import astropy.units as u
import itur
from langchain_core.tools import tool

@tool
def calculate_itu_attenuations(lat: float, lon: float, distance_km: float, frequency_ghz: float, availability_pct: float) -> str:
    """
    Computes Free Space Loss (FSL), Atmospheric Attenuation, and Rain Attenuation for a specific location.
    Requires: latitude, longitude, link distance (km), frequency (GHz), and target availability (%).
    """
    f = frequency_ghz * u.GHz
    d = distance_km * u.km
    p = (100.0 - availability_pct) # Exceedance probability

    # Free Space Loss (FSL)
    fsl_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 92.45
    
    # Atmospheric Gaseous Attenuation (ITU-R P.676)
    # Assumes standard atmospheric conditions (15°C, 1013.25 hPa, 7.5 g/m^3 water vapor)
    T = 15 * u.deg_C
    P = 1013.25 * u.hPa
    rho = 7.5 * (u.g / u.m**3)
    gamma_gas = itur.models.itu676.gaseous_attenuation_terrestrial_path(d, f, P, rho, T)
    
    # Rain Attenuation (ITU-R P.530 / P.838 / P.837)
    # Note: ITU-R models expect horizontal/vertical polarization. We default to Horizontal (0 deg) for worst-case.
    el = 0 * u.deg # Terrestrial link
    tau = 0 * u.deg # Horizontal polarization
    a_rain = itur.models.itu530.rain_attenuation(lat, lon, d, f, el, p, tau)
    
    return (f"FSL: {fsl_db:.2f} dB\n"
            f"Atmospheric Gas Attenuation: {gamma_gas.value:.2f} dB\n"
            f"Rain Attenuation ({availability_pct}% availability): {a_rain.value:.2f} dB")

@tool
def calculate_antenna_specs(frequency_ghz: float, diameter: float, unit: str) -> str:
    """
    Calculates parabolic dish antenna gain (dBi) and Half-Power Beamwidth (degrees).
    Requires: frequency (GHz), diameter, and unit ('cm' or 'ft').
    """
    # Convert diameter to meters
    if unit.lower() == 'ft':
        diameter_m = diameter * 0.3048
    elif unit.lower() == 'cm':
        diameter_m = diameter / 100.0
    else:
        diameter_m = diameter # assume meters fallback

    # Constants
    c = 299792458.0 # m/s
    f_hz = frequency_ghz * 1e9
    wavelength_m = c / f_hz
    efficiency = 0.60 # Standard 60% aperture efficiency
    
    # Gain Calculation
    gain_linear = efficiency * ((math.pi * diameter_m) / wavelength_m)**2
    gain_dbi = 10 * math.log10(gain_linear)
    
    # Beamwidth Calculation
    beamwidth_deg = 70 * (wavelength_m / diameter_m)
    
    return f"Antenna Gain: {gain_dbi:.2f} dBi, Half-Power Beamwidth: {beamwidth_deg:.2f}°"

@tool
def calculate_rx_threshold(qam_level: int, channel_bw_mhz: float, frequency_ghz: float) -> str:
    """
    Estimates the receive level threshold (dBm) based on the Shannon limit, coding rate (0.88), 
    and dynamically scaled Noise Figure.
    Requires: QAM level (e.g., 1024), channel bandwidth (MHz), and frequency (GHz).
    """
    # 1. Spectral Efficiency (bits/symbol)
    m = math.log2(qam_level)
    coding_rate = 0.88
    spectral_efficiency = m * coding_rate
    
    # 2. Required SNR based on Shannon Limit + margins
    # Shannon: C/B = log2(1 + SNR) -> SNR_linear = 2^(C/B) - 1
    snr_linear = (2 ** spectral_efficiency) - 1
    snr_req_db = 10 * math.log10(snr_linear) + 1.6 + 4.0
    
    # 3. Dynamic Noise Figure (Interpolated 3dB to 7dB between 6GHz and 84GHz)
    # Bound the frequencies to avoid negative or extreme NFs
    freq_bounded = max(6.0, min(84.0, frequency_ghz))
    nf_db = 3.0 + ((7.0 - 3.0) / (84.0 - 6.0)) * (freq_bounded - 6.0)
    
    # 4. Thermal Noise and Final Threshold
    thermal_noise_dbm = -174.0 + 10 * math.log10(channel_bw_mhz * 1e6)
    rx_threshold_dbm = thermal_noise_dbm + nf_db + snr_req_db
    
    return (f"Rx Threshold: {rx_threshold_dbm:.2f} dBm\n"
            f"(Calculated using {nf_db:.2f}dB NF and {snr_req_db:.2f}dB Required SNR)")

@tool
def calculate_link_availability(qam_level: int, channel_bw_mhz: float, distance_km: float, 
                              antenna_diameter_m: float, frequency_ghz: float, tx_power_dbm: float, 
                              lat: float, lon: float) -> str:
    """
    Calculates expected link availability (%) by running a full link budget.
    Requires: QAM, BW (MHz), distance (km), antenna diameter (m), frequency (GHz), Tx Power (dBm), lat, and lon.
    """
    # 1. Get Tx/Rx parameters
    rx_thresh_str = calculate_rx_threshold.invoke({"qam_level": qam_level, "channel_bw_mhz": channel_bw_mhz, "frequency_ghz": frequency_ghz})
    rx_threshold = float(rx_thresh_str.split(":")[1].split("dBm")[0].strip())
    
    ant_specs_str = calculate_antenna_specs.invoke({"frequency_ghz": frequency_ghz, "diameter": antenna_diameter_m, "unit": "m"})
    gain_dbi = float(ant_specs_str.split("Gain:")[1].split("dBi")[0].strip())
    
    # 2. Get clear sky propagation
    f = frequency_ghz * u.GHz
    d = distance_km * u.km
    fsl_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_ghz) + 92.45
    gamma_gas = itur.models.itu676.gaseous_attenuation_terrestrial_path(d, f, 1013.25*u.hPa, 7.5*(u.g/u.m**3), 15*u.deg_C).value
    
    # 3. Calculate Fade Margin
    # Assumes identical antennas at both ends: Tx + Gain(Tx) - FSL - Gas + Gain(Rx)
    rx_clear_sky = tx_power_dbm + (2 * gain_dbi) - fsl_db - gamma_gas
    fade_margin = rx_clear_sky - rx_threshold
    
    if fade_margin <= 0:
        return f"Link fails in clear sky. Fade margin is {fade_margin:.2f} dB."

    # 4. Iteratively solve for Availability using ITU-R P.530
    def fade_difference(p_exceedance):
        # We find the exceedance probability 'p' where rain attenuation exactly equals our fade margin
        a_rain = itur.models.itu530.rain_attenuation(lat, lon, d, f, 0*u.deg, p_exceedance, 0*u.deg).value
        return a_rain - fade_margin

    try:
        # Search for the root between 0.00001% (high availability) and 50% exceedance
        result = root_scalar(fade_difference, bracket=[1e-5, 50], method='brentq')
        availability = 100.0 - result.root
        
        return (f"Clear Sky Rx Level: {rx_clear_sky:.2f} dBm\n"
                f"Fade Margin: {fade_margin:.2f} dB\n"
                f"Expected Availability: {availability:.5f}%")
    except ValueError:
        return f"Fade Margin is {fade_margin:.2f} dB. Availability > 99.9999% (exceeds standard ITU bounds)."


# --- Main Search & Telecom Chat Interface ---
st.title("💬 IP50EX/CX/GP/20N-Series Agent")
st.caption(f"Active Agent Reasoning Engine: **{selected_model_label}**")

# 1. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display existing chat messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Chat Input (Pins to the bottom of the screen)
query = st.chat_input("Ask a technical/telecom engineering question:")

if query:
    # Immediately display the user's question in the chat UI
    with st.chat_message("user"):
        st.markdown(query)
    
    # Save user query to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Generate and display the agent's response
    with st.chat_message("assistant"):
        with st.spinner(f"Agent is analyzing your request using {selected_model_id}..."):
            try:
                # 1. Instantiate the LLM
                chat_llm = get_chat_llm(selected_model_id)
                
                # 2. Convert Pinecone RAG into a Tool
                user_filter = ROLE_FILTERS[st.session_state['user_role']]
                retriever = vectorstore.as_retriever(search_kwargs={"filter": user_filter, "k": 4})
                
                rag_tool = create_retriever_tool(
                    retriever=retriever,
                    name="search_hardware_manuals",
                    description="Searches authorized technical manuals. Use this to answer questions about hardware specs, configurations, and IP50/VX-Series documentation."
                )
                
                # 3. Bundle the tools together
                tools = [calculate_rain_attenuation, rag_tool]
                
                # 4. Define the Agent's Core Instructions
                system_prompt = SystemMessage(content="""You are an expert telecom and wireless hardware engineering assistant.
                You have tools available to you. 
                - If the user asks for a calculation (like rain attenuation), use the calculation tool.
                - If the user asks for hardware specs or documentation, use the document search tool.
                - Do NOT guess hardware specs or math equations. Rely strictly on your tools.
                
                At the very end of your final answer, provide exactly 2 highly relevant follow-up questions the user could ask. Format them as a bulleted list under the bold heading: **Suggested Follow-up Questions:**
                """)
                
                # 5. Build the LangGraph Agent
                agent = create_react_agent(chat_llm, tools, prompt=system_prompt)
                
                # 6. Format the chat history for LangGraph
                # LangGraph expects a simple list of tuples for standard conversational memory
                chat_history = []
                for m in st.session_state.messages:
                    chat_history.append((m["role"], m["content"]))
                        
                # 7. Execute the Agent
                # The agent autonomously loops between the LLM and the tools until it has a final answer
                response = agent.invoke({"messages": chat_history})
                
                # 8. Extract and display the final answer
                # The final answer from the agent is always the content of the very last message in the list
                final_answer = response["messages"][-1].content
                st.markdown(final_answer)
                
                # Save the agent's response to the Streamlit UI history
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                error_msg = f"❌ Agent Execution Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
