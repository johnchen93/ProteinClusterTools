import pandas as pd
from sklearn.cluster import MiniBatchKMeans, BisectingKMeans
import multiprocessing as mp
import numpy as np

def Kmeans(embeddings, n_clusters, outlabel):
    # convert embeddings dict to numpy ndarray
    ids=np.array(list(embeddings.keys()))
    X=np.array(list(embeddings.values()))

    # run kmeans, using minibatch kmeans for speed
    if type(n_clusters)==int:
        n_clusters=[n_clusters]

    clusters={}
    for k in n_clusters:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        clusters[k]=kmeans.labels_
    
    # make dataframe and write to file
    df=pd.DataFrame.from_dict(clusters)
    df['id']=ids
    df=df[['id']+n_clusters]
    out_name=f'{outlabel}_kmeans.csv'
    df=df.sort_values('id', key=lambda x: x.astype(int))
    df.to_csv(out_name, index=False)
    return {'kmeans': out_name}

def KMeansBisecting(embeddings, n_clusters, ids, out_dir='', out_prefix=''):

    # run kmeans, using minibatch kmeans for speed
    if type(n_clusters)==int:
        n_clusters=[n_clusters]

    # clusters={}
    for k in n_clusters:
        print(f'Running bisecting kmeans with {k} clusters')
        kmeans = BisectingKMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        # make a dataframe, with id and cluster assignments
        
        pd.DataFrame({'id':ids, 'cluster':kmeans.labels_}).to_csv(f'{out_dir}{out_prefix}_kmeans_bisecting_{k}.csv', index=False)

### Bisecting KMeans with tree structure
### This is a modified version of the BisectingKMeans class from sklearn
### The only change is that the tree structure is kept, so that the cluster assignments can be traced back to the original data

from sklearn.base import _fit_context
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
# from sklearn.utils._param_validation import Integral, Interval, StrOptions
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_random_state
from sklearn.cluster._k_means_common import _inertia_dense, _inertia_sparse
from sklearn.cluster._kmeans import (
    _BaseKMeans,
    _kmeans_single_elkan,
    _kmeans_single_lloyd,
    _labels_inertia_threadpool_limit,
)
from sklearn.cluster._bisect_k_means import _BisectingTree
import scipy.sparse as sp

class BisectingKMeansKeepTree(BisectingKMeans):

    # subclass of BisectingKMeans that keeps the tree structure
    def __init__(self, n_clusters=2, *, max_iter=100, tol=1e-4, random_state=None, verbose=0):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose)
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Only change is to not erase the tree labels"""
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        self._random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        if self.algorithm == "lloyd" or self.n_clusters == 1:
            self._kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            self._kmeans_single = _kmeans_single_elkan

        # Subtract of mean of X for more accurate distance computations
        if not sp.issparse(X):
            self._X_mean = X.mean(axis=0)
            X -= self._X_mean

        # Initialize the hierarchical clusters tree
        self._bisecting_tree = _BisectingTree(
            indices=np.arange(X.shape[0]),
            center=X.mean(axis=0),
            score=0,
        )

        x_squared_norms = row_norms(X, squared=True)

        for _ in range(self.n_clusters - 1):
            # Chose cluster to bisect
            cluster_to_bisect = self._bisecting_tree.get_cluster_to_bisect()

            # Split this cluster into 2 subclusters
            self._bisect(X, x_squared_norms, sample_weight, cluster_to_bisect)

        # Aggregate final labels and centers from the bisecting tree
        self.labels_ = np.full(X.shape[0], -1, dtype=np.int32)
        self.cluster_centers_ = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        for i, cluster_node in enumerate(self._bisecting_tree.iter_leaves()):
            self.labels_[cluster_node.indices] = i
            self.cluster_centers_[i] = cluster_node.center
            cluster_node.label = i  # label final clusters for future prediction
            # cluster_node.indices = None  # release memory # Only change

        # Restore original data
        if not sp.issparse(X):
            X += self._X_mean
            self.cluster_centers_ += self._X_mean

        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense
        self.inertia_ = _inertia(
            X, sample_weight, self.cluster_centers_, self.labels_, self._n_threads
        )

        self._n_features_out = self.cluster_centers_.shape[0]

        return self

### Helpers for tree structure
import pandas as pd
# from sklearn.cluster import MiniBatchKMeans, BisectingKMeans
# import sklearn.cluster as cluster
# import importlib
# importlib.reload(cluster)
import numpy as np
# import pickle
# from Bio import Phylo
# import PCA
# from sklearn.decomposition import PCA

# make tree structure
def to_newick(node):
    if node is None:
        return ""
    
    # If the node is a leaf, return its name
    if node.left is None and node.right is None:
        label = int(node.label)+1
        return label
    
    # Recursively process left and right children
    left_newick = to_newick(node.left)
    right_newick = to_newick(node.right)
    
    left_dist = abs(node.score - node.left.score)
    right_dist = abs(node.score - node.right.score)

    # Combine the parts with parentheses
    if left_newick and right_newick:
        return f"({left_newick}:{left_dist},{right_newick}:{right_dist})"
    elif left_newick:
        return f"({left_newick}:{left_dist})"
    elif right_newick:
        return f"({right_newick}:{right_dist})"
    
    return ""

def tree_to_newick(root):
    newick_str = to_newick(root)
    if newick_str:
        return newick_str + ";"
    return ""

def get_leaf_index_mapping(tree):
    """
    Generate a mapping of leaf names to their indices and vice versa.
    """
    leaves = tree.get_terminals()
    leaf_names = [leaf.name for leaf in leaves]
    leaf_to_index = {name: i  for i, name in enumerate(leaf_names)}
    index_to_leaf = {i : name for i, name in enumerate(leaf_names)}
    return leaf_to_index, index_to_leaf

def get_merge_info(node, min_dists):
    
    # assumes input only has internal nodes, that each node is numbered by index, that tree traversal is post order

    ### (left for reference, very slow) get distance as minimum distance of leaves between the two nodes
    ### using min distance gives the single linkage
    # print('calculating distance...')
    # left_dist=min([tree.distance(node, leaf) for leaf in node.clades[0].get_terminals()])
    # right_dist=min([tree.distance(node, leaf) for leaf in node.clades[1].get_terminals()])

    left= node.clades[0]
    right= node.clades[1]

    # assumes distance traversal is leaves first
    left_dist=(0 if left.is_terminal() else min_dists[left.name])+left.branch_length
    right_dist=(0 if right.is_terminal() else min_dists[right.name])+right.branch_length

    min_dists[node.name]=min(left_dist, right_dist)

    distance=left_dist + right_dist

    n= len(node.get_terminals())

    # Append linkage information
    return [left.name, right.name, distance, n, node]

import time
from tqdm import tqdm
def compute_linkage_from_tree(tree):
    """
    Compute the linkage matrix directly from a tree with indexed leaves.
    """
    print('Setting up...')
    leaf_to_index, _ = get_leaf_index_mapping(tree)
    leaves = tree.get_terminals()
    num_leaves = len(leaves)
    linkage_list = []
    index_to_node={}

    
    clade_list=[]
    # traverse tree and gather all internal clades
    i=num_leaves
    # node_indices={} 
    for clade in tree.find_clades():
        if clade.is_terminal():
            # node_indices[clade]=leaf_to_index[clade.name]
            # rename to leaf index
            clade.name=leaf_to_index[clade.name]
            continue
        clade.name=i # give clade an arbitrary index
        i+=1
        # node_indices[clade]=clade.name
        index_to_node[clade.name]=clade
        clade_list.append(clade)
    # index_to_node={v:k for k, v in node_indices.items()}
    # invert clade list to start closest to leaves
    clade_list=clade_list[::-1]

    min_dists={}
    
    # tstart=time.time() # debug
    # print('traversing tree...')
    # traverse(tree.root)
    for clade in tqdm(clade_list, desc='traversing tree'):
        merge_info=get_merge_info(clade, min_dists)
        if merge_info is not None:
            linkage_list.append(merge_info)
    # print('time taken:', time.time()-tstart)
    
    # Convert to numpy array for scipy linkage function
    linkage_list=sorted(linkage_list, key=lambda x: x[2])
    nodes=[x[4] for x in linkage_list]
    linkage_list=[[x[0], x[1], x[2], x[3]] for x in linkage_list]
    Z = np.array(linkage_list)
    # sort by distance
    Z=Z[Z[:,2].argsort()]

    # relabel any node with n>num_leaves by order of appearance
    c=num_leaves
    ordered_node_index={}
    # first get ordered indices of nodes
    for node in nodes:
        ordered_node_index[node]=c
        c+=1
    # then replace internal node indices with new indices
    for row in Z:
        for i in [0, 1]:
            if row[i]>=num_leaves:
                row[i]=ordered_node_index[index_to_node[row[i]]]
    
    return Z, [leaf.name for leaf in tree.get_terminals()]

# def convert_linkage_to_clusters(Z, num_clusters):
#     """
#     Convert linkage matrix to clusters using fcluster.
#     """
#     cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')
#     return cluster_labels