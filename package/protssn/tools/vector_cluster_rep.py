import pandas as pd
import numpy as np

def ClusterVectorRepresentative(clusters, level, vector_dict, target_clusters):
    cluster_ids=clusters.groupby(level)['id'].apply(list).to_dict()

    cluster_reps={}
    for c, ids in cluster_ids.items():
        if c not in target_clusters:
            continue
        members=np.array([vector_dict[x] for x in ids])
        ids=np.array(ids)
        representative=VectorRepresentative(members, ids)
        cluster_reps[c]=representative
        
    rep_df=pd.DataFrame(cluster_reps, index=[0]).T.reset_index().rename(columns={'index':'cluster', 0:'top_hit'})
    return rep_df

def VectorRepresentative(members, ids):
    mean=np.mean(members, axis=0)
    dist=np.linalg.norm(members-mean, axis=1)
    idx=np.argmin(dist)
    return ids[idx]