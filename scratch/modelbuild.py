#!/usr/bin/python3

################################################
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import textwrap
workDir = "."

# to read and manipulate the data
import pandas as pd
import numpy as np
pd.set_option('max_colwidth', None)    # setting column to the maximum column width as per the data

# Deep Learning library
import torch

# to load transformer models
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# to split the data
from sklearn.model_selection import train_test_split

# to compute performance metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Function to download the model from the Hugging Face model hub
from huggingface_hub import hf_hub_download

# Importing the Llama class from the llama_cpp module
from llama_cpp import Llama

# Importing the json library

# for persisting models to disk
import joblib

################################################
# Read KB text from file
kb_file="/ConjurCloudDocs.csv"
# loading data into a pandas dataframe
knowledge_base = pd.read_csv(workDir+kb_file)
knowledge_base.drop(columns=['Unnamed: 0'],inplace=True)
# creating a copy of the data
data = knowledge_base.copy()

################################################
'''
# Remove noise from KB text
def removeNoise(x, startstr, stopstr):
  startidx = x.find(startstr)
  stopidx = x.find(stopstr) + len(stopstr)
  return x[0:startidx] + x[stopidx:]

startstr = "CyberArk Docs Table of Contents"
stopstr = "Docs\n"
data['Text'] = data['Text'].apply(lambda x: removeNoise(x,startstr, stopstr))
startstr = "Was this topic helpful?"
stopstr = "]"
data['Text'] = data['Text'].apply(lambda x: removeNoise(x,startstr, stopstr))
'''
################################################
# Create KB embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# there are many other models to choose from too!
# https://www.sbert.net/docs/pretrained_models.html
# https://huggingface.co/spaces/mteb/leaderboard
# model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoding the dataset
embedding_matrix = model.encode(data['Text'], device=device, show_progress_bar=True)

################################################
# Query the KB

#-----------------------------------------------------------
# defining a function to find the top k similar entries for a given query
def top_k_similar_entries(embedding_matrix,query_text,k):
    # encoding the query text
    query_embedding = model.encode(query_text)

    # calculating the cosine similarity between the query vector and all other encoded vectors of our dataset
    score_vector = np.dot(embedding_matrix,query_embedding)

    # sorting the scores in descending order and choosing the first k
    top_k_indices = np.argsort(score_vector)[::-1][:k]

    # returning the corresponding reviews
    return data.loc[list(top_k_indices), 'Url']
#-----------------------------------------------------------

# defining the query text
query_text = "least privilege policy"

# displaying the top 5 similar sentences
top_k_reviews = top_k_similar_entries(embedding_matrix,query_text,5)

for i in top_k_reviews:
    print(i, end="\n\n")

################################################
# Build LLM

model_path = "../langchain/"
model_basename = "llama-2-13b-chat.Q5_K_M.gguf" # the model is in gguf format
n_gpu_layers = -1   # -1 to move all to GPU.
n_ctx = 4096        # Context window
n_batch = 512       # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
lcpp_llm = Llama(
    model_path=model_path+model_basename,
    n_cpus=-1,
    n_batch=n_batch,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=n_gpu_layers,  # Change this value based on your model and your GPU VRAM pool.
    n_ctx=n_ctx,  # Context window
)
################################################
# find the top KB entry for a given query
def top_KB_entry(embedding_matrix,query_text):
    # encoding the query text
    query_embedding = model.encode(query_text)

    # calculating the cosine similarity between the query vector and all other encoded vectors of our dataset
    score_vector = np.dot(embedding_matrix,query_embedding)

    top_index = np.argsort(score_vector)[::-1][0]

    # returning indices of the top entries
    return top_index

def generate_llama_response(instruction, context):

    # System message explicitly instructing not to include the context text
    system_message = """
    [INST] <<SYS>>
    <</SYS>>
    {}[/INST]
    """.format(instruction)

    # Combine user_prompt and system_message to create the prompt
    prompt = f"{context}\n{system_message}"

    # Generate a response from the LLaMA model
    response = lcpp_llm(
        prompt=prompt,
        max_tokens=4096,
        temperature=0,
        top_p=0.95,
        repeat_penalty=1.2,
        top_k=50,
        stop=['INST'],
        echo=False,
        seed=42,
    )

    # Extract the response
    response_text = response["choices"][0]["text"]
    return response_text


# defining the query text
query_text = "configure jenkins for authn-jwt"

max_words = 1000
# get the most similar KB entries and concatenate them as bullet points
top_entry = top_KB_entry(embedding_matrix,query_text)
top_kb_entry_text = data.iloc[top_entry]['Text']
kb_word_count = len(top_kb_entry_text.split())
print("Word count:", kb_word_count, "\nText:\n", top_kb_entry_text)
if kb_word_count > max_words:
  print("Trimming")
  kb_entry = ' '.join(top_kb_entry_text.split()[:max_words])
else:
  kb_entry = top_kb_entry_text

# defining the instructions for the model
ins1 = """
    You are an AI providing courteous, accurate, step by step advice regarding the above topic.
    Be concise.
    Use fhe following information in your response:
    - {}
""".format(kb_entry)

advice = generate_llama_response(ins1, query_text)
textwrap.wrap(advice, 80)