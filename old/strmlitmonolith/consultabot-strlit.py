#!/usr/bin/python3

# Import necessary libraries
import streamlit as st
from PIL import Image
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# ============================================
import logging
logfile = "./logs/consultabot.log"
loglevel = logging.DEBUG
logfmode = 'w'                # w = overwrite, a = append
logging.basicConfig(filename=logfile, encoding='utf-8', level=loglevel, filemode=logfmode)

# ============================================
# Setup Streamlit page configuration
im = Image.open('sricon.png')
st.set_page_config(page_title=' 🤖ConsultaBot!🧠', layout='wide', page_icon = im)
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "just_sent" not in st.session_state:
    st.session_state["just_sent"] = False
if "temp" not in st.session_state:
    st.session_state["temp"] = ""

with st.sidebar:
    st.markdown("---")
    st.markdown("# About")
    st.markdown(
       "ConsultaBot is a RAG LLM chatbot. "
       "It can answer questions about Conjur Cloud."
            )
    st.markdown(
       "This tool is a work in progress. "
            )

# Set up the Streamlit app layout
st.title("🤖 ConsultaBot! 🧠")
#st.subheader(" Powered by 🦜 LangChain + LLama + Streamlit")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Add a button to start a new chat
#st.sidebar.button("New Chat", on_click = new_chat, type='primary')

#####################################################
def clear_text():
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""

#####################################################
# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input", 
                            placeholder="ConsultaBot is here to help! Ask me anything ...", 
                            on_change=clear_text,    
                            label_visibility='hidden')
    input_text = st.session_state["temp"]
    return input_text

#####################################################
# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
#with st.sidebar.expander("🛠️ ", expanded=False):
#    # Option to preview memory store
#    if st.checkbox("Preview memory store"):
#        with st.expander("Memory-Store", expanded=False):
#            st.session_state.entity_memory.store
#    # Option to preview memory buffer
#    if st.checkbox("Preview memory buffer"):
#        with st.expander("Buffer-Store", expanded=False):
#            st.session_state.entity_memory.buffer
#    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
#    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

################################################
# Load vector store from disk
def loadVectorStore():
    VECTORSTORE_PATH="./"
    VECTORSTORE_FILE="LangChain_FAISS"

    print("\nRead vectorstore from disk...")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    _vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        hf_embedder,
        VECTORSTORE_FILE,
        allow_dangerous_deserialization=True,  # we trust it because we created it
    )
    return _vectorstore

################################################
# Load LLM from disk
def loadLLM():
    print("\nLoading LLM from disk...")

    #model_path = "./llama-2-13b-chat.Q5_K_M.gguf")
    #model_path = "./Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf"
    model_path = "./Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    temp=0              # stick to the facts
    n_gpu_layers = -1   # -1 to move all to GPU.
    n_ctx = 4096        # Context window
    n_batch = 512       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    _llm = LlamaCpp(
        model_path=model_path,
        temperature=temp,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return _llm

################################################
# Conversational QA chain (chat history)
def createConversationChain(_llm, _vectorstore, _st):
    # Create a ConversationEntityMemory object if not already created
    K = 100
    if 'entity_memory' not in _st.session_state:
            _st.session_state.entity_memory = ConversationEntityMemory(llm=_llm, k=K ) 

    _chat = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_vectorstore.as_retriever(),
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=_st.session_state.entity_memory
    )
    return _chat

################################################
# One-off QA chain (no memory)
def createQAChain(_llm, _vectorstore):
    _chat = RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=_vectorstore.as_retriever()
    )
    return _chat

# MAIN =======================================
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = loadVectorStore()
vectorstore = st.session_state["vectorstore"]

if "llm" not in st.session_state:
    st.session_state["llm"] = loadLLM()
llm = st.session_state["llm"]

if "chat" not in st.session_state:
    st.session_state["chat"] = createQAChain(llm, vectorstore)
    #st.session_state["chat"] = createConversationChain(llm, vectorstore, st)
chat = st.session_state["chat"]

user_input = get_text()
if user_input:
    chat = st.session_state["chat"]
    #output = chat.run(input=user_input)
    output = chat.invoke({"query": user_input}).get("result")
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output) 

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("chat", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
                            
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session