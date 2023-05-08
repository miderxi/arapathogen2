from Bio import SeqIO
import numpy as np
import word2vec
import pickle

#1. load sequences of uniprot
seqs_list =[(i.id,str(i.seq)) for i in  SeqIO.parse("../../data/sequences/ara_and_eff.fasta","fasta")]

#2.build word with k-mer
with open("./swis.txt","w") as f:
    for idx,(t_id,t_seq) in enumerate(seqs_list):
        seq=[]
        for  i in range(len(t_seq)-1):
            seq.append(t_seq[i:i+2])
        seq = " ".join(seq)
        seq = seq.lower()
        seq = seq.encode("ascii","ignore")
        f.write(f"_*{idx} {seq}\n")


# save index file, where index is the number and value is the id.
with open("./swis_id2num.txt","w") as f:
    for idx,(t_id,t_seq) in enumerate(seqs_list):
        f.write(f"{t_id} {idx}\n")

#3.trainning doc2vec model

word2vec.doc2vec('./swis.txt', './swis_doc2vec-vectors2.bin', cbow=0, size=128, window=10, negative=5,
                 hs=0, sample='1e-4', threads=12, iter_=20, min_count=1, binary=True, verbose=True)


