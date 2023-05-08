import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
import sys
import pickle


def dpc(k,seq):
    aminos = "AGVCDEFILPHNQWKRMSTY"
    dpc_dict = {i+j:0 for i in aminos for j in aminos}
    seq = "".join([i for i in seq.upper() if i in aminos])

    for i in range(len(seq)-1):
        dpc_dict[seq[i:i+2]] += 1
    values = np.array(list(dpc_dict.values())) / (len(seq)-1)
    return k, values

def main():
    encode_func = dpc

    in_file = sys.argv[1]  if len(sys.argv) > 1 else "../../data/sequences/ara_and_eff.fasta"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "./ara_and_eff_DPC_400.pkl" 

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


