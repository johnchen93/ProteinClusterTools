from collections import defaultdict
import re
import subprocess
import tempfile
import os
from Bio import SeqIO
# import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import pandas as pd

# single letter list of amino acids, sorted by type
aa_list = ['H', 'K', 'R',                # (+)
           'D', 'E',                     # (-)
           'C', 'M', 'N', 'Q', 'S', 'T', # Polar-neutral
           'A', 'G', 'I', 'L', 'P', 'V', # Non-polar
           'F', 'W', 'Y',                # Aromatic
           '*']
_non_aa_re = re.compile(f'[^{"".join(aa_list)}]+')

def ReadFasta(file, id_index=None):
    seqs={}
    for record in SeqIO.parse(file, "fasta"):
        id=record.id
        if id_index is not None:
            id=id.split('|')[id_index]
        if _non_aa_re.search(str(record.seq)):
            print(f'Warning: {id} contains non-amino acid characters. Excluded from analysis.')
            continue
        seqs[id]=str(record.seq)
    return seqs

def FilterFastas(seqs, filter, id_map=None):
    out={}
    for k,v in seqs.items():
        if k in filter:
            id=id_map[k] if id_map is not None else k
            out[id]=v
    return out

def WriteFasta(file, seqs):
    with open(file,'w') as f:
        for k,v in seqs.items():
            
            f.write(f'>{k}\n{v}\n')

from multiprocessing import Pool
import shutil

def process_cluster(cluster_data):
    cluster, ids, fa, id_map, tmpdir = cluster_data

    fa_path = os.path.join(tmpdir, f'cluster_{cluster}.fa')
    msa_path = os.path.join(tmpdir, f'cluster_{cluster}.aln')
    hmm_path = os.path.join(tmpdir, f'cluster_{cluster}.hmm')
    hits_path = os.path.join(tmpdir, f'cluster_{cluster}.hits')

    seqs = FilterFastas(fa, ids, id_map)
    if seqs == {}:
        print(f'Warning: Cluster {cluster} is empty. Skipping.')
        return None
    WriteFasta(fa_path, seqs)
    
    # Run external commands
    subprocess.call(['mafft', fa_path], stdout=open(msa_path, 'w'), stderr=subprocess.DEVNULL)
    subprocess.call(['hmmbuild', hmm_path, msa_path], stdout=subprocess.DEVNULL)
    subprocess.call(['hmmsearch', '--tblout', hits_path, hmm_path, fa_path], stdout=subprocess.DEVNULL)

    # Process hits to find the top hit
    top_hit = None
    with open(hits_path) as hits:
        for line in hits:
            if not line.startswith('#'):
                top_hit = line.split()[0]
                break
    
    return cluster, msa_path, hmm_path, hits_path, top_hit

def ParseClusterLevel(cluster_file, level, fasta_file, target_clusters, zip_path, id_index=None, num_processes=None, id_map=None):
    '''
    cluster_file: csv file with columns 'id' and potentially multiple 'levels'
    level: the level to use for clustering
    fasta_file: fasta file with sequences, expected to have the same headers as the 'id' column in cluster_file
                - if using the cleaned fasta, ids will be in integers
    target_clusters: list of clusters to analyze
    zip_path: path to save the zip file
    id_index: index of the id in the fasta header assuming splitting on '|'
    num_processes: number of processes to use for parallelization, default uses all
    id_map: A filename, or dataframe (with 'id' and 'header' columns, where 'id' is the current label, and 'header' is the desired mapping) for changing the output headers from the original, e.g. from indices to original headers
    '''

    fa = ReadFasta(fasta_file, id_index)

    if type(cluster_file)==str:
        df=pd.read_csv(cluster_file, dtype={'id':str})
    else:
        df=cluster_file

    # collect id and level columns and convert to dictionary, group by level
    clusters=df.groupby(level)['id'].apply(list).to_dict()
    # print(clusters)

    if id_map is not None:
        if type(id_map)==str:
            id_map=pd.read_csv(id_map, dtype=str).set_index('id')['header'].to_dict()
        else:
            id_map=id_map.set_index('id')['header'].to_dict()

    # Prepare data for multiprocessing
    cluster_data_list = [(cluster, ids, fa, id_map, tempfile.mkdtemp()) for cluster, ids in clusters.items() if cluster in target_clusters]

    # Run analysis on each cluster in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_cluster, cluster_data_list)

    # make sure output directory exists
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    # Write results to zip
    with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zip_file:
        top = {}
        for result in results:
            if result is not None:
                cluster, msa_path, hmm_path, hits_path, top_hit = result
                top[cluster] = top_hit
                zip_file.write(msa_path, f'{cluster}/cluster_{cluster}.aln')
                zip_file.write(hmm_path, f'{cluster}/cluster_{cluster}.hmm')
                zip_file.write(hits_path, f'{cluster}/cluster_{cluster}.hits')
                # Clean up temporary files
                os.remove(msa_path)
                os.remove(hmm_path)
                os.remove(hits_path)

        # Add summary files to the zip
        top_table_str = 'cluster,top_hit\n' + '\n'.join([f'{k},{v}' for k, v in top.items()])
        zip_file.writestr('cluster_top_hits.csv', top_table_str)
        top_ids=[x for x in top.values() if x is not None]

        if id_map is not None:
            look_up=dict(zip(id_map.values(),id_map.keys()))
            top_ids=[look_up[x] for x in top_ids]
            # print(top_ids)
        zip_file.writestr('cluster_top_hits.fasta', ''.join([f'>{k}\n{v}\n' for k, v in FilterFastas(fa, top_ids, id_map).items()]))

    # Clean up temporary directories
    for _, _, _, _, tmpdir in cluster_data_list:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_file')
    parser.add_argument('fasta_file')
    parser.add_argument('--fasta_id_index', '-fi', type=int, help='Index of ID in fasta header')
    parser.add_argument('target_clusters')
    parser.add_argument('zip_path')
    args=parser.parse_args()

    target_clusters=set()
    with open(args.target_clusters) as f:
        for line in f:
            cluster=line.strip()
            if cluster=='':
                continue
            if cluster not in target_clusters:
                target_clusters.add(cluster)

    ParseClusterLevel(args.cluster_file, args.fasta_file, target_clusters, args.zip_path, args.fasta_id_index)