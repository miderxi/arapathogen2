import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
import sys
import pickle


def cksaap(prot_id,seq,k=3):
    aminos = "AGVCDEFILPHNQWKRMSTY"
    seq = ''.join([i for i in seq.upper() if i in aminos])
    ckdpc =  {str(idx)+i+j:0 for idx in range(k) for i in aminos for j in aminos}

    for idx in range(k):
        for i in range(len(seq)-idx-1):
            ckdpc[str(idx)+seq[i]+seq[i+idx+1]]+=1

    values = np.array(list(ckdpc.values()))/ (len(seq)-k)
    return prot_id, values


def main():
    encode_func = cksaap

    in_file = sys.argv[1]  if len(sys.argv) > 1 else "../../data/sequences/ara_and_eff.fasta"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "./ara_and_eff_CKSAAP.pkl" 

    seqs = {i.id:str(i.seq) for i in SeqIO.parse(in_file,"fasta")}

    #compute
    pool = Pool(12)
    pool_list = []
    for k,v in seqs.items() :
        pool_list.append(pool.apply_async(encode_func,[k,v]))
    pool.close()
    pool.join()

    #extract
    encode_list = [i.get() for i in pool_list]
    encode_dict = {a:np.array(b,np.float32) for a,b in encode_list}

    #save
    with open(out_file,"wb") as f:
        pickle.dump(encode_dict,f)


main()


