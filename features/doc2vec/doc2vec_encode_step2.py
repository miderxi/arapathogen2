from Bio import SeqIO
import numpy as np
import word2vec
import pickle

#4.extract feature
# prepare candicate id for extracte
ids = [i.id  for i in  SeqIO.parse("../../data/sequences/ara_and_eff.fasta","fasta")]
ids_set = set(ids)

#5 read number to id file
ids2num = dict()
for prot_idx, doc2vec_idx in np.genfromtxt("./swis_id2num.txt",str):  #left is protein id,right is index of doc2vec model embeding vector.
    if prot_idx not in ids2num.keys() and prot_idx in ids_set:
        ids2num[prot_idx] = doc2vec_idx

#6 load model
model = word2vec.load('./swis_doc2vec-vectors2.bin')

#7 extract embeding vector
total_features = {}
for prot_idx,doc2vec_index in ids2num.items():
    total_features[prot_idx] = model[f"_*{doc2vec_index}"]

#8 save to disk.
with open("./ara_and_eff_doc2vec_128.pkl","wb") as f:
    pickle.dump(total_features,f)


