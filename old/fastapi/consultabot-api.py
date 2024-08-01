#!/usr/bin/python3

import logging
from fastapi import FastAPI, Body

logfile = "./logs/consultabot.log"
loglevel = logging.DEBUG
logfmode = 'w'                # w = overwrite, a = append

# MAIN ============================================
logging.basicConfig(filename=logfile, encoding='utf-8', level=loglevel, filemode=logfmode)


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
print("\nLoading LLM from disk...")

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

#model_path = "./llama-2-13b-chat.Q5_K_M.gguf")
#model_path = "./Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf"
model_path = "./Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = -1   # -1 to move all to GPU.
n_ctx = 4096        # Context window
n_batch = 512       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_ctx,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

################################################
print("\nConnect LLM to vector store retriever...")
from langchain.chains import RetrievalQA

# create retriever object
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

################################################
# see:
# - https://fastapi.tiangolo.com/tutorial/body/
# - https://stackoverflow.com/questions/64057445/fastapi-post-does-not-recognize-my-parameter
from pydantic import BaseModel

class Query(BaseModel):
    data: str



def getBotResponse(query):
  return qa_chain.invoke({"query": query}).get("result")

app = FastAPI()

# Allow access from localhost
# see: https://fastapi.tiangolo.com/tutorial/cors/
from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]
'''
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://10.0.9.167",
    "http://10.0.9.167:3000",
    "http://99.79.38.81",
    "http://99.79.38.81:3000",
]
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
def _getBotResponse(query: Query):
    return getBotResponse(query.data)



