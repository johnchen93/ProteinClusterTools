import pickle
from scipy.cluster.hierarchy import fcluster, to_tree#, linkage
from fastcluster import linkage
from Bio import Phylo
import numpy as np
import pandas as pd
import os

def HierarchicalClustering(ids, embeddings, out_dir='', out_prefix='', method='weighted', metric='cosine'):

    # run hierarchical clustering
    Z = linkage(embeddings, method=method, metric=metric)

    # save linkage matrix
    outfile=f'{out_prefix}_linkage_matrix.pkl'
    idsfile=f'{out_prefix}_linkage_ids.pkl'
    with open(os.path.join(out_dir, outfile), 'wb') as f:
        pickle.dump(Z, f)
    with open(os.path.join(out_dir, idsfile), 'wb') as f:
        pickle.dump(ids, f)
    return {'linkage_matrix': outfile, 'linkage_ids': idsfile}

def FlattenHierarchy(linkage_matrix, linkage_ids, cutoffs, out_dir='', out_prefix='', criterion='maxclust'):
    # load linkage matrix
    with open(linkage_matrix, 'rb') as f:
        Z = pickle.load(f)
    # load ids
    with open(linkage_ids, 'rb') as f:
        ids = pickle.load(f)

    # flatten hierarchy
    for cutoff in cutoffs:
        clu=fcluster(Z, cutoff, criterion=criterion)
        # make dataframe and write to file
        df=pd.DataFrame({'id':ids, 'cluster':clu})
        df.to_csv(os.path.join(out_dir, f'{out_prefix}_hierarchical_flat_{cutoff}.csv'), index=False)

def linkage_to_newick(linkage_matrix, leaf_names):
    def get_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{parentdist - node.dist}{newick}"
        else:
            if len(newick) > 0:
                newick = f"):{node.dist}{newick}"
            else:
                newick = ");"
            newick = get_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = get_newick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
            newick = "(%s" % (newick)
            return newick

    tree = to_tree(linkage_matrix, False)
    newick = get_newick(tree, "", tree.dist, leaf_names)
    return newick

def NameInternalNodes(treefile):
    tree=Phylo.read(treefile, 'newick')
    # travel through tree and name internal nodes
    node=0
    for clade in tree.get_nonterminals(order='preorder'):
        
        clade.name=f'node_{node}'
        node+=1
    write_newick(tree, treefile)

def write_newick(tree, out_file):
    def _write_node(node):
        node_name = f"{node.name}:{node.branch_length}"
        if node.is_terminal():
            return node_name
        else:
            children = ','.join(_write_node(child) for child in node.clades)
            return f"({children}){node_name}"

    newick_string = _write_node(tree.root) + ";"
    with open(out_file, "w") as file_handle:
        file_handle.write(newick_string + "\n")