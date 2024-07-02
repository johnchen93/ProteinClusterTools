import os, shutil
import pickle
from .esm_extract import create_parser, run
import torch
from Bio import SeqIO

def Embed(fasta, out_dir='', prefix='', tmp_dir='tmp/esm/', tok_per_batch=30000, cont_run=False):

    # clear tmp_dir if it already exists
    if os.path.exists(tmp_dir) and not cont_run:
        shutil.rmtree(tmp_dir)

    if cont_run and os.path.exists(tmp_dir):
        # read all file names in tmp_dir
        files=os.listdir(tmp_dir)
        # get basename without path and extensions
        ids=set([f.split('.')[0] for f in files if f.endswith('.pt')])

        # read in fasta file, get ids of stuff yet to be processed
        records=SeqIO.parse(fasta, 'fasta')
        yet_to_process=[x for x in records if x.id not in ids]

        # write fasta to tmp_dir
        fasta_in=os.path.join(tmp_dir, 'tmp.fasta')
        with open(fasta_in, 'w') as f:
            SeqIO.write(yet_to_process, f, 'fasta')
        print(f'Continuing embedding remaining {len(yet_to_process)} sequences')
    else:
        fasta_in=fasta

    args_str=['esm1b_t33_650M_UR50S', fasta_in, tmp_dir, '--toks_per_batch' , str(tok_per_batch), '--include', "mean"]
    parser=create_parser()
    args=parser.parse_args(args_str)
    run(args)

    embed_file=f'{prefix}_embeddings.pkl'
    embeddings={}
    # open torch models and extract embeddings
    for f in os.listdir(tmp_dir):
        if not f.endswith('.pt'):
            continue
        model=torch.load(os.path.join(tmp_dir, f))
        id=f.split('.')[0]
        embeddings[id]=model['mean_representations'][33]
    # pickle embeddings
    with open(os.path.join(out_dir,embed_file), 'wb') as f:
        pickle.dump(embeddings, f)

    # remove tmp_dir
    shutil.rmtree(tmp_dir) 

    return {'embeddings': embed_file}