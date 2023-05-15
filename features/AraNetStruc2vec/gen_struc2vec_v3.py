import networkx as nx
#from GraphEmbedding.ge.models import Struc2Vec #old version
from ge import Struc2Vec
import pickle

def run1():
    G = nx.read_edgelist('./Intact_TAIR_protein_interaction.txt',
                        create_using=nx.DiGraph(),    
                        nodetype=None,    
                        data=[('weight',int)])
    
    #for fast compute,walk_length 40,num_walks=200 train very cost time
    model = Struc2Vec(G,walk_length=10, 
                      num_walks=200, 
                      workers=12, 
                      opt3_num_layers=4,
                      verbose=40)

    model.train(embed_size=256, 
                window_size = 10, 
                iter = 50, 
                workers=12)

    embeddings = model.get_embeddings()# get embedding vectors

    with open("./AraNetStruc2vec_256_v3.pkl","wb") as f:
        pickle.dump(embeddings,f)

run1()

