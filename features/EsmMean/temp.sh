mkdir ./tmp_esm_out/
#python extract.py esm1b_t33_650M_UR50S input.fasta ./tmp_esm_out/ --include mean --repr_layers 33 --truncation_seq_length 2000
python extract2.py  esm2_t36_3B_UR50D  ../../data/sequences/only_need.fasta ./tesm_out/ \
    --include mean  \
    --repr_layers 36  \
    --truncation_seq_length 4000  \
    --save_file  \
    ./ara_and_eff_only_need_esm2.pkl
