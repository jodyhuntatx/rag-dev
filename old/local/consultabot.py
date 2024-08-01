#!/usr/bin/python3

# Get environment variables for HF token
import os, sys
hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
if hf_token is None:
  print("HUGGINGFACE_TOKEN env var must be set to valid Hugging Face token.")
  sys.exit(-1)

################################################
VECTORSTORE_PATH="./"
VECTORSTORE_FILE="LangChain_FAISS"

print("\nRead vectorstore from disk...")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embedder = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    hf_embedder,
    VECTORSTORE_FILE,
    allow_dangerous_deserialization=True,  # we trust it because we created it
)

################################################
print("\nTest query the vector store...")

question = "What is Conjur Edge?"
docs = vectorstore.similarity_search(question)
print(docs)

################################################
print("\nConnecting to LLM...")
def get_hf_hosted_model():
    print("\nGet connection to model hosted by HuggingFace...")
    from langchain_community.llms import HuggingFaceHub
    # repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
    repo_id = "google/flan-t5-large"
    _llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
    )
    return _llm

def get_ollama_model():
    print("\nGet connection to Ollama model...")
    from langchain_community.llms import Ollama

    _llm = Ollama(model="llama2")
    # llm = Ollama(model="mistral")
    return _llm

def load_model_from_disk(model_path):
    from langchain_community.llms import LlamaCpp
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = -1   # -1 to move all to GPU.
    n_ctx = 4096        # Context window
    n_batch = 512       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    
    _llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    
    return _llm

#llm = get_ollama_model()
#llm = get_hf_hosted_model()
llm = load_model_from_disk("./llama-2-13b-chat.Q5_K_M.gguf")

################################################
print("\nConnect LLM to vector store retriever...")
from langchain.chains import RetrievalQA

# create retriever object
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

################################################
print("\nTest one-off LLM response generation...")
question = "What platforms does Conjur Edge support?"
print(qa_chain.invoke({"query": question}))

################################################
print("\nAdd stateful conversation support...")
# create memory object
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

retriever = vectorstore.as_retriever()
chat = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

################################################
print("\nDemonstration...")
print(chat.invoke("How can I setup an Edge node?"))
print(chat.invoke("an edge node is the same thing as an edge server."))
