from Bio import SeqIO
import numpy as np

all_seqs = {_.id:str(_.seq) for _ in SeqIO.parse("./ara_and_eff.fasta","fasta")}
only_need_ids = set(
        np.genfromtxt("../10folds_C1223/C1_C2h_C2p_and_C3.txt",str)[:,:2].ravel()
        )

with open("./only_need.fasta","w") as f:
    for k,v in all_seqs.items():
        if k in only_need_ids:
            f.write(f">{k}\n{v}\n") 
