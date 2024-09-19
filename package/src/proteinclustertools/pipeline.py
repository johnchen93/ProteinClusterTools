from .tools.cluster_merging import SetsToDataFrame, iterative_merge
from .tools.cluster_stream import ClusterStreamMulti
from .tools.hierarchical_clustering import FlattenHierarchy, HierarchicalClustering, linkage_to_newick, NameInternalNodes
from .tools.mmseqs_wrapper import RunMMseqs
from .tools.summarize_distribution import read_and_sample, calculate_percentiles, plot_distribution
from .tools.esm_wrapper import Embed
from .tools.sanitize_fasta_headers import sanitize_fasta_headers
from .tools.kmeans import KMeansBisecting, BisectingKMeansKeepTree, compute_linkage_from_tree, tree_to_newick

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from umap import UMAP
import shutil
from scipy.cluster.hierarchy import fcluster
from Bio import Phylo

def StateFile(out_label):
    return f'{out_label}_states.pkl'

def SaveStates(states, state_file):
    # read in state file
    old_states=pickle.load(open(state_file, 'rb')) if os.path.exists(state_file) else {}
    # update states
    old_states.update(states)
    with open(state_file, 'wb') as f:
        pickle.dump(old_states, f)

def PathToFile(directory, path):
    return os.path.join(directory, path)

def LoadEmbeddings(directory, states, filter_path=None):
    embeddings_file=PathToFile(directory, states['embeddings'])
    embeddings=pickle.load(open(embeddings_file, 'rb'))
    if filter_path:
        # read in filter as a set
        print('Filtering embeddings by id')
        filter_set=set()
        with open(filter_path, 'r') as f:
            for line in f:
                filter_set.add(line.strip())
        # filter embeddings
        embeddings={k:v for k,v in embeddings.items() if k in filter_set}
    # convert embeddings from pytorch tensor to numpy array
    ids=list(embeddings.keys())
    embeddings=np.array(list(embeddings.values()))

    return ids, embeddings

def ReduceVectors(embeddings, keep_features):
    # get length of 1st vector
    vector_length=len(embeddings[0])

    if keep_features < vector_length and keep_features > 0:
        print('Reducing feature dimensions with PCA')
        pca = PCA(n_components=keep_features)
        return pca.fit_transform(embeddings)
    else:
        return embeddings

import argparse

parser = argparse.ArgumentParser(description='Pipeline for analyzing protein families using unsupervised clustering. Uses either homology or vector embeddings to cluster sequences.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

ga=parser.add_argument_group('General arguments')
ga.add_argument('-d', '-directory', help='Output directory for files', default='out/')
ga.add_argument('-p', '-prefix', help='Prefix for output files', default='ssn')
ga.add_argument('-fa', '-fasta', help='Fasta file to analyze. Only needs to be given once.', default=None)

hom=parser.add_argument_group('Homology based method')
# mmseqs all-by-all
hom.add_argument('-A','-all_by_all', action='store_true', help='Run mmseqs all-by-all search')
# mmseqs all-by-all clustering
hom.add_argument('-MC','-mmseqs_cluster', action='store_true', help='Cluster mmseqs results')
hom.add_argument('-P', '-cluster_percentiles', action='store_true', help='Use percentiles of mmseqs results to do clustering.')
hom.add_argument('-cc', '-cluster_cutoffs', nargs='+', type=float, help='Manually set cutoffs for clustering. Is ignored if -P is set.')
hom.add_argument('-cluster_lines', type=int, help='Number of lines to read at once, adjust for memory consumption.', default=10_000_000)
hom.add_argument('-f', '-filter', help='List of IDs to filter for. Can be used for mmseqs clustering, kmeans, and UMAP.')
hom.add_argument('-RMC', '-redo_mmseqs_clustering', action='store_true', help='Redo steps for mmseqs clustering.')
hom.add_argument('-cluster_jobs', type=int, help='Number of jobs to use for clustering', default=None)

# esm embedding
emb=parser.add_argument_group('Vector embedding')
emb.add_argument('-E','-embed_vectors', action='store_true', help='Embed vectors')
emb.add_argument('-CE','-continue_embed', action='store_true', help='Continue embedding vectors, in case of interrupted run.')
emb.add_argument('-t', '-tok_per_batch', type=int, help='Number of tokens per batch for embedding. Reduce if running out of memory.', default=30_000)

vec=parser.add_argument_group('Vector based methods')
# general vector
vec.add_argument('-KF', '-keep_features', type=int, help='Number of features to use in vector based methods.', default=30)
# Kmeans clustering
vec.add_argument('-KM', '-kmeans', action='store_true', help='Run kmeans clustering')
vec.add_argument('-K', '-kmeans_clusters', nargs='+', type=int, help='Number of kmeans clusters. Also used for setting cluster count when flattening hierachical clusters.', default=[])
vec.add_argument('-FKM', '-flatten_kmeans', action='store_true', help='Subsect kmeans clusters based on max_k tree')
vec.add_argument('-max_k', type=int, help='Max K to further subdivide into clusters')
# Hierarchical clustering
vec.add_argument('-HC', '-hierarchical_clustering', action='store_true', help='Run hierarchical clustering')
vec.add_argument('-CT', '-convert_hierarchical_tree', action='store_true', help='Convert hierarchical tree to newick format')
# Flatten hierarchical clustering
vec.add_argument('-FHC', '-flatten_hierarchical_clustering', action='store_true', help='Flatten hierarchical clustering')
# UMAP
vec.add_argument('-U', '-umap', action='store_true', help='Run UMAP')
args=parser.parse_args()

##### Settings to human readable #####
# This is where the output data will go, includes the prefix for the output files, and the output directory
out_directory=args.d
out_prefix=args.p

all_by_all=False or args.A # run mmseqs all-by-all search

embed_vectors=False or args.E # embed vectors
continue_embed=False or args.CE # continue embedding vectors, use if previous run was somehow interrupted
tok_per_batch=args.t # number of tokens per batch for embedding

mmseqs_cluster=False or args.MC # cluster mmseqs results
use_percentiles=False or args.P # use percentiles of mmseqs results to do clustering
cluster_cutoffs=args.cc # cutoffs for clustering

run_kmeans=False or args.KM # run kmeans clustering
kmeans_clusters=args.K # number of kmeans clusters
keep_features=args.KF # number of features to use in kmeans clustering
flatten_kmeans=False or args.FKM # subsect kmeans clusters based on max_k tree
flatten_kmeans_max_k=args.max_k # max K to further subdivide into clusters

run_hierarchical_clustering=False or args.HC # run hierarchical clustering
convert_hierarchical_tree=False or args.CT # convert hierarchical tree to newick format
flatten_hierarchical_clustering=False or args.FHC # flatten hierarchical clustering

run_umap=False or args.U # run UMAP
####################

# make sure the output directory exists
os.makedirs(out_directory, exist_ok=True)

# make a file to keep track of outputs
state_file=StateFile(PathToFile(out_directory, out_prefix))
if os.path.exists(state_file):
    with open(state_file, 'rb') as f:
        states=pickle.load(f)
else:
    states={}

# the fasta file to be analyzed
fasta=args.fa

if 'cleaned_fasta' not in states:
    # clean up fasta headers by converting them to an index
    fasta_info=sanitize_fasta_headers(fasta, out_directory, out_prefix)

    # save the state
    print(f'files produced in {out_directory}:', fasta_info)
    states.update(fasta_info)
    SaveStates(states, state_file)

# make the path to the file, for other analyses
cleaned_fasta=PathToFile(out_directory, states['cleaned_fasta'])
print('cleaned fasta is in:', cleaned_fasta)

# Run mmseqs
if all_by_all:
    RunMMseqs_out=RunMMseqs(cleaned_fasta, out_directory, out_prefix)

    print('Updates to state:', RunMMseqs_out)
    states.update(RunMMseqs_out)
    SaveStates(states, state_file)

if 'mmseqs_result' in states and 'mmseqs_percentiles' not in states:
    # # define paths to data
    mmseqs_result=PathToFile(out_directory, states['mmseqs_result'])
    image_out=PathToFile(out_directory, f'{out_prefix}_bits.png')
    target_metric='bits' # metric stored in the mmseqs_result table

    # helper functions that finds the 10, 25, 50, 75, and 90 percentiles of the pairwise edge table, and plots the distribution of the data
    data = read_and_sample(mmseqs_result, 'bits', max_points=20_000_000) # subsample if there are more than X entries
    percentiles = calculate_percentiles(data, [.1, .25, .5, .75, .9]) # provide target precentiles as fractions
    print('percentiles:', percentiles)
    plot_distribution(data, 'bits', plot_out=image_out, show=False)

    states['mmseqs_percentiles']=percentiles
    SaveStates(states, state_file)

# cluster mmseqs results
if mmseqs_cluster:

    if use_percentiles:
        cutoffs=list(states['mmseqs_percentiles'].values())
    else:
        cutoffs=cluster_cutoffs

    if 'mmseqs_cluster_cutoffs' not in states:
        states['mmseqs_cluster_cutoffs']={}
    for cutoff in cutoffs:
        if cutoff not in states['mmseqs_cluster_cutoffs']:
            states['mmseqs_cluster_cutoffs'][cutoff]='started'
    SaveStates(states, state_file)

    # cluster the mmseqs results in chunks
    edge_table_path=PathToFile(out_directory, states['mmseqs_result'])
    cluster_path=os.path.join(out_directory, 'mmseqs_clustering/')
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    use_cutoffs=[x for x in cutoffs if x not in states['mmseqs_cluster_cutoffs'] or states['mmseqs_cluster_cutoffs'][x] == 'started'] if not args.RMC else cutoffs
    if len(use_cutoffs) == 0:
        print('All mmseqs clusterings are already done. Proceeding to cluster merging.')
    else:
        print('Clustering mmseqs results with cutoffs:', use_cutoffs)
        ClusterStreamMulti(
            edge_table_path, 
            use_cutoffs, 
            query=0, target=1, score=2, 
            has_header=True, field_separator="\t", 
            # label=out_prefix, 
            directory=cluster_path, 
            chunksize=args.cluster_lines, n_jobs=None, filter_path=args.f)
    
    for cutoff in use_cutoffs:
        if states['mmseqs_cluster_cutoffs'][cutoff] == 'started':
            states['mmseqs_cluster_cutoffs'][cutoff]='chunked'
    SaveStates(states, state_file)

    # merge the chunked clusterings into a single file
    merge_cutoffs=[x for x in cutoffs if (states['mmseqs_cluster_cutoffs'][x] == 'chunked' or args.RMC)]
    print('Merging mmseqs clusters:', merge_cutoffs)
    for cutoff in merge_cutoffs:
        folder=str(cutoff)
        print('Processing folder:',folder)
        folder_path=os.path.join(cluster_path, folder)
        files = [x for x in os.listdir(folder_path) if x.endswith('.pkl')]
        datasets = [os.path.join(folder_path, x) for x in files]
        merged_dataset = iterative_merge(datasets, max_datasets=args.cluster_jobs)
        # save the merged dataset
        out_file = os.path.join(cluster_path, f'{out_prefix}_mmseqs_clusters_{cutoff}.csv')
        SetsToDataFrame(merged_dataset).to_csv(out_file, index=False)
        print('Saved:', out_file)
        if states['mmseqs_cluster_cutoffs'][cutoff] == 'chunked':
            states['mmseqs_cluster_cutoffs'][cutoff]='done'
        SaveStates(states, state_file)

        # remove the folder with the chunked clusterings
        shutil.rmtree(folder_path)

if embed_vectors:
    embed_out=Embed(cleaned_fasta, out_directory, out_prefix, tok_per_batch=tok_per_batch, cont_run=continue_embed) # change tok_per_batch lower if you run out of GPU memory

    print('Updates to state:', embed_out)
    states.update(embed_out)
    SaveStates(states, state_file)

if run_kmeans:
    # kmeans_path=os.path.join(out_directory, 'kmeans/max_k/')
    # # make sure the output directory exists
    # os.makedirs(kmeans_path, exist_ok=True)

    print('Loading embeddings for kmeans')
    ids, embeddings=LoadEmbeddings(out_directory, states, args.f)

    # get length of 1st vector
    embeddings=ReduceVectors(embeddings, keep_features)

    # run kmeans
    print('Running kmeans')
    # KMeansBisecting(embeddings, kmeans_clusters, ids, kmeans_path, out_prefix)

    # save each cluster file as a 'max K' file, that is then fed to a lower K clustering
    new_states={}
    for k in kmeans_clusters:
        kmeans = BisectingKMeansKeepTree(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)

        # paths
        label_file=f'kmeans/max_k/k{k}_labels.csv'
        tree_file=f'kmeans/max_k/k{k}_tree.nwk'
        os.makedirs(os.path.join(out_directory, 'kmeans/max_k/'), exist_ok=True)


        labels=pd.DataFrame({'id':ids, 'label':kmeans.labels_+1})
        labels.to_csv(os.path.join(out_directory, label_file), index=False)
        # make tree structure
        tree=tree_to_newick(kmeans._bisecting_tree)

        # write tree to file
        with open(os.path.join(out_directory, tree_file), 'w') as f:
            f.write(tree)
        
        new_states[k]={'labels':label_file, 'tree':tree_file}
    states['max_k']=new_states
    SaveStates(states, state_file)

if flatten_kmeans:
    # check if max_k exists
    if flatten_kmeans_max_k is None:
        print('Please provide a max K to subsect kmeans clusters. (use -max_k)')
    elif flatten_kmeans_max_k not in states['max_k']:
        print(f'Max K not found in states. There are {len(states["max_k"])} max K values processed: {list(states["max_k"].keys())}')
 
    # read tree
    tree_file=PathToFile(out_directory, states['max_k'][flatten_kmeans_max_k]['tree'])
    tree=Phylo.read(tree_file, 'newick')
    linkage_matrix, leaf_names = compute_linkage_from_tree(tree)
    labels_file=PathToFile(out_directory, states['max_k'][flatten_kmeans_max_k]['labels'])
    labels=pd.read_csv(labels_file)

    # make sure the output directory exists
    k_out_path=os.path.join(out_directory, f'kmeans/flattened_{flatten_kmeans_max_k}/')
    # print(k_out_path)
    os.makedirs(k_out_path, exist_ok=True)

    # create smaller clusters
    for smaller_k in kmeans_clusters:
        if smaller_k > flatten_kmeans_max_k:
            continue
        print('Flattening kmeans clusters to:', smaller_k)
        # Apply fcluster to get cluster labels
        cluster_labels = fcluster(linkage_matrix, smaller_k, criterion='maxclust')

        # write labels to file
        leaf_labels=pd.DataFrame({'label':leaf_names, 'cluster':cluster_labels})

        # merge on labels
        merged=labels.merge(leaf_labels, on='label')
        # write to file
        # print(os.path.join(k_out_path, f'k{smaller_k}_clusters.csv'))
        merged[['id','cluster']].to_csv(os.path.join(k_out_path, f'{out_prefix}_kmeans_bisecting_{smaller_k}.csv'), index=False)

if run_umap:
    # load vector embeddings
    print('Loading embeddings for UMAP')
    ids, embeddings=LoadEmbeddings(out_directory, states, args.f)

    embeddings=ReduceVectors(embeddings, keep_features)
    
    # perform UMAP
    print('Performing UMAP')
    umap=UMAP(n_components=2)
    umap_embeddings=umap.fit_transform(embeddings)

    # make dataframe
    umap_df=pd.DataFrame(umap_embeddings, columns=['x', 'y'])
    # add sequence ids
    umap_df['id']=ids

    # save and save to state
    umap_file=out_prefix+'_umap.csv'
    umap_df.to_csv(PathToFile(out_directory, umap_file), index=False)
    states['umap']=umap_file
    SaveStates(states, state_file)

if run_hierarchical_clustering:
   
    ids, embeddings=LoadEmbeddings(out_directory, states, args.f)

    embeddings=ReduceVectors(embeddings, keep_features)

    # run hierarchical clustering
    hc_out=HierarchicalClustering(ids, embeddings, out_directory, out_prefix)
    print('Updates to state:', hc_out)
    states.update(hc_out)

    SaveStates(states, state_file)

if convert_hierarchical_tree and 'linkage_matrix' in states:
    # see if distance matrix exists
    matrix=pickle.load(open(PathToFile(out_directory, states['linkage_matrix']), 'rb'))
    ids=pickle.load(open(PathToFile(out_directory, states['linkage_ids']), 'rb'))

    # convert linkage matrix to newick format
    newick=linkage_to_newick(matrix, ids)
    # save newick format
    outfile=f'{out_prefix}_hc_tree.nwk'
    outpath=os.path.join(out_directory, outfile)
    with open(outpath, 'w') as f:
        f.write(newick)
    
    # rename internal nodes
    NameInternalNodes(outpath)

    states.update({'hc_tree':outfile})
    SaveStates(states, state_file)

if flatten_hierarchical_clustering:
    hc_path=os.path.join(out_directory, 'hierarchical_clustering/')
    # make sure the output directory exists
    os.makedirs(hc_path, exist_ok=True)

    # load linkage matrix
    linkage_matrix=PathToFile(out_directory, states['linkage_matrix'])
    linkage_ids=PathToFile(out_directory, states['linkage_ids'])
    cutoffs=kmeans_clusters

    # flatten hierarchy
    FlattenHierarchy(linkage_matrix, linkage_ids, cutoffs, hc_path, out_prefix)