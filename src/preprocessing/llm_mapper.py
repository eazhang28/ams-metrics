# from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from gpt4all import GPT4All
import hdbscan
import pandas as pd
import numpy as np
import torch
import pickle
import sys
import time
import ast

def import_data(filepath, cache = False) -> np.array: 
    # note: impelment caching
    df = pd.read_csv(filepath)
    data = df.iloc[:,[10]]
    data_np = data.to_numpy()
    print(data)
    return data_np

def data_to_docs(data) -> list:
    documents = set()
    for x in data:
        documents.add(str(x[0]))
    documents = list(documents)
    return documents

# model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
# print("The model BAAI/bge-small-en-v1.5 is ready to use.")
# model = SentenceTransformer('all-MiniLM-L6-v2')

def gen_map(documents, cache=False) -> list:
    # note: implement caching
    in_str = ", ".join(x for x in documents)
    in_str = "Please group and assign canonical labels to the following list of titles:\n" + in_str
    in_str += "\nReturn an output as a list of canoncial labels in the format a python list. Return only the list and nothing else."
    model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    with model.chat_session():

        out_str = model.generate(in_str)
        # if cache == True:
        print(f"mapping:\n{out_str}")

    quit()
    return str_to_list(out_str)
    
def str_to_list(str_map) -> list:
    return ast.literal_eval(str_map)

def embedding_clustering(documents, model):
    embed_list = model.encode(documents, normalize_embeddings=True)

    # embed_list = list(model.embed(documents))
    # print(np.shape(embed_list))

    cl_model = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
    data_labels = list(cl_model.fit_predict(embed_list))

    label_map = dict()
    num_label_stack = list(set(data_labels))
    num_label_stack = list(int(x) for x in num_label_stack)
    i = 0
    while len(num_label_stack) >= 1 and i < len(data_labels):
        if num_label_stack[0] in data_labels:
            label_map[num_label_stack[0]]=documents[data_labels.index(num_label_stack[0])]
            num_label_stack.pop(0)
        i += 1

    print(label_map)
    return label_map

if __name__ == '__main__':
    data = import_data("data/raw/_survey_data_DGData12__202503061442.csv")
    documents = data_to_docs(data)
    print(documents)
    if len(sys.argv) == 3:
        if sys.argv[2] == "-l":
            gen_map(documents)
        elif sys.argv[2] == "-e":
            embedding_clustering(documents, SentenceTransformer('all-MiniLM-L6-v2'))
    else:
        print("Use a flag!")