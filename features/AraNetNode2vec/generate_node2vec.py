import sys
import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import pickle
import numpy as np
#1. generate ara net features

# load graph
def main(input_file,output_file):
    graph = nx.read_edgelist(input_file, create_using=nx.DiGraph(), nodetype=None)

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=256, walk_length=40, num_walks=400, workers=12)  # Use temp_folder for big graphs

    # Embed nodes
    # Any keywords acceptable by gensim.Word2Vec can be passed, 
    #`dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    model = node2vec.fit(window=20, min_count=1, batch_words=8, workers=16)  

    # Save embeddings for later use
    model.wv.save_word2vec_format("../AraNetStruc2vec/Intact_TAIR_protein_interaction.txt")

    AraNet_node2vec_dict = {line[0]:np.array(line[1:],np.float32) 
                        for line in np.genfromtxt("./ara_node2vec_feature.txt",str,delimiter=" ",skip_header=1)}
    with open(output_file,"wb") as f:
        pickle.dump(AraNet_node2vec_dict,f)


script_path = sys.path[0]
input_file  = f"../AraNetStruc2vec/Intact_TAIR_protein_interaction.txt"
output_file = f"./AraNet_node2vec_256.pkl"

main(input_file=input_file, output_file=output_file)

