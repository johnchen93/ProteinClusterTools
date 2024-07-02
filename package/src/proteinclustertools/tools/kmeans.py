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
