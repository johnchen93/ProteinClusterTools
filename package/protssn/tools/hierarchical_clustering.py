import pickle
from scipy.cluster.hierarchy import fcluster#, linkage
from fastcluster import linkage
import numpy as np
import pandas as pd
import os

def HierarchicalClustering(ids, embeddings, out_dir='', out_prefix='', method='weighted', metric='cosine'):
    # convert embeddings dict to numpy ndarray
    # ids=np.array(list(embeddings.keys()))
    # X=np.array(list(embeddings.values()))

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
    # clusters={}
    for cutoff in cutoffs:
        clu=fcluster(Z, cutoff, criterion=criterion)
        # make dataframe and write to file
        df=pd.DataFrame({'id':ids, 'cluster':clu})
        df.to_csv(os.path.join(out_dir, f'{out_prefix}_hierarchical_flat_{cutoff}.csv'), index=False)

        # clusters[cutoff]=fcluster(Z, cutoff, criterion=criterion)
    # # make dataframe and write to file
    # df=pd.DataFrame.from_dict(clusters)
    # ids=pickle.load(open(linkage_ids, 'rb'))
    # df['id']=ids
    # df=df[['id']+cutoffs]
    # out_name=f'{out_prefix}_hierarchical_flat.csv'
    # df=df.sort_values('id', key=lambda x: x.astype(int))
    # df.to_csv(os.path.join(out_dir, out_name), index=False)

    # return {'hierarchical_flat': out_name}