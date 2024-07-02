from collections import defaultdict
from multiprocessing import Pool
import os
import pickle
import pandas as pd
import time
import argparse


def GetConnectedComponents(df, all_ids=[], id1_col="id1", id2_col="id2_list", singles=[], silence=False):
    """
    expect df to be a dataframe with two columns
    the first column is the id of a sequence, and the second column is a list of ids of sequences that it is connected to (excluding self)
    singles is a list of all ids
    """
    sets = defaultdict(set)
    singles = []
    tested = {}
    n = 0
    ids = df[id1_col].values
    members = df[id2_col].values
    for i in range(len(ids)):
        id = ids[i]
        mems = members[i]
        if len(mems) == 1 and id in mems:
            singles.append(id)
            continue

        cur_set = set(list(mems) + [id])
        for m in cur_set:
            if m not in tested:
                tested[m] = True
        found = []
        for key, s in sets.items():
            if cur_set & s:
                found.append(key)
        if len(found) == 0:
            sets[n] = cur_set
            n += 1
            continue
        join_key = min(found)
        for key in [k for k in found if k != join_key]:
            sets[join_key] |= sets.pop(key, None)
        sets[join_key] |= cur_set
    if not silence:
        print(f"{len(sets)} clusters with {sum([len(x) for x in sets.values()])} sequences")
    added = 0
    for id in singles:
        # map the single clusters
        if id not in tested:
            sets[n] = set([id])
            tested[id] = True
            n += 1
            added += 1
            continue
    if not silence:
        print(f"{added} single nodes added")
    not_connected = [x for x in all_ids if x not in tested]
    for id in not_connected:
        sets[n] = set([id])
        n += 1
    if not silence:
        print(f"{len(not_connected)} unconnected nodes added")
    return sets

### Multiprocessing version
def ClusterChunkWrapper(args):
    return ClusterChunk(*args)

def ClusterChunk(chunk, chunk_id, cutoffs, id1_col, id2_col, score_col, out_dir, filter=None):
    if filter is not None:
        chunk = chunk[(chunk[id1_col].isin(filter)) & (chunk[id2_col].isin(filter))]
    
    for cutoff in cutoffs:
        file_dir=f"{out_dir}/{cutoff}"
        # make sure the directory exists
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        # print(f"---cutoff: {cutoff}")
        chunk = chunk[(chunk[score_col] >= cutoff) | (chunk[id1_col] == chunk[id2_col])]

        df = chunk.groupby([id1_col])[id2_col].agg(set).reset_index()
        clusters = GetConnectedComponents(df, id1_col=id1_col, id2_col=id2_col, silence=True)
        pickle.dump(clusters, open(f"{file_dir}/{chunk_id}.pkl", "wb"))
        print(f"Chunk {chunk_id} cutoff {cutoff} done")

def ChunkInput(iterator, *args):
    for index, chunk in enumerate(iterator):
        # make list or tuple with index, chunk and rest of args
        yield chunk, index, *args

def ClusterStreamMulti(
    edge_table,
    cutoffs,
    query=0,
    target=1,
    score=11,
    label="cluster",
    directory=".",
    chunksize=10_000_000,
    has_header=False,
    field_separator="\t",
    filter_path=None,
    n_jobs=None):
    '''Multiprocessing version of cluster_stream
        Only handles the per-chunk clustering, not the final organization of clusters into a single file.

        Requires cluster_merging to be run after this function
    '''
    if has_header:
        header = 0
        for chunk in pd.read_csv(
            edge_table, delimiter=field_separator, chunksize=10, header=header
        ):
            qcol = chunk.columns[query]
            tcol = chunk.columns[target]
            scol = chunk.columns[score]
            break
    else:
        header = None
        qcol = query
        tcol = target
        scol = score

    # make sure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    df_iterator = pd.read_csv(
        edge_table, delimiter=field_separator, chunksize=chunksize, header=header, usecols=[qcol, tcol, scol]
    )

    filter_set = set(pd.read_csv(filter_path, header=None)[0]) if filter_path else None

    with Pool(n_jobs) as pool:

        pool.imap_unordered(ClusterChunkWrapper, ChunkInput(df_iterator, cutoffs, qcol, tcol, scol, directory, filter_set))
        pool.close()
        pool.join()

### Old version
def Cluster(
    df_iterator, cutoffs, id1_col, id2_col, score_col, label, outdir, filter=None
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    start = time.time()
    i = 0
    sets = defaultdict(dict)
    c = 0
    cutoffs = sorted(cutoffs)
    for chunk in df_iterator:
        i += len(chunk)
        if filter is not None:
            chunk = chunk[(chunk[id1_col].isin(filter)) & (chunk[id2_col].isin(filter))]

        for cutoff in cutoffs:
            print(f"---cutoff: {cutoff}")
            chunk = chunk[(chunk[score_col] >= cutoff) | (chunk[id1_col] == chunk[id2_col])]

            df = chunk.groupby([id1_col])[id2_col].agg(set).reset_index()
            clusters = GetConnectedComponents(df, id1_col=id1_col, id2_col=id2_col)
            for cluster in clusters.values():
                found = []
                for cid, s in sets[cutoff].items():
                    if set.intersection(cluster, s):
                        found.append(cid)
                if not found:
                    sets[cutoff][c] = cluster
                    c += 1
                else:
                    min_id = min(found)
                    for cid in found:
                        cluster |= sets[cutoff].pop(cid, None)
                    sets[cutoff][min_id] = cluster

        print(i, f"\n{time.time()-start:.2f}s\n")

    # organize all cut-off data into dataframes
    clusters={}
    for cutoff in cutoffs:
        i=0
        data={}
        for _, s in sets[cutoff].items():
            for x in s:
                data[x]=i
            i += 1    
        clusters[cutoff]=pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index':'id', 0:cutoff}) 

    # save all clusters in same file
    file=f"{label}_clusters.csv"
    file_path=os.path.join(outdir, file)
    table=None
    if os.path.exists(file_path):
        table=pd.read_csv(file_path)
        print(table.columns)
        table=table.drop(columns=[str(x) for x in cutoffs], errors='ignore')

    for cutoff in cutoffs:
        if table is None:
            table=clusters[cutoff]
        else:
            table=table.merge(clusters[cutoff], on='id', how='outer')
    table.to_csv(file_path, index=False) 

    return {'clusters':file}

def cluster_stream(
    edge_table,
    cutoffs,
    query=0,
    target=1,
    score=11,
    label="cluster",
    directory=".",
    chunksize=10_000_000,
    has_header=False,
    field_separator="\t",
    filter_path=None,
):
    '''Main cluster function for non-multiprocessing version'''
    if has_header:
        header = 0
        for chunk in pd.read_csv(
            edge_table, delimiter=field_separator, chunksize=10, header=header
        ):
            qcol = chunk.columns[query]
            tcol = chunk.columns[target]
            scol = chunk.columns[score]
            break
    else:
        header = None
        qcol = query
        tcol = target
        scol = score

    df_iterator = pd.read_csv(
        edge_table, delimiter=field_separator, chunksize=chunksize, header=header
    )
    filter_set = set(pd.read_csv(filter_path, header=None)[0]) if filter_path else None

    return Cluster(df_iterator, cutoffs, qcol, tcol, scol, label, directory, filter_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "-edge_table", help="Path to edge file", required=True)
    parser.add_argument(
        "-c",
        "-cutoffs",
        type=float,
        nargs="+",
        help="List of cutoffs for analysis.",
        required=True,
    )
    parser.add_argument(
        "-q",
        "-query",
        type=int,
        default=0,
        help="Index of column in edge table for QUERY id.",
    )
    parser.add_argument(
        "-t",
        "-target",
        type=int,
        default=1,
        help="Index of column in edge table for TARGET id.",
    )
    parser.add_argument(
        "-s",
        "-score",
        type=int,
        default=11,
        help="Index of column in edge table for SCORE.",
    )
    parser.add_argument(
        "-l", "-label", default="cluster", help="Outlabel, suffixes will be added."
    )
    parser.add_argument("-d", "-dir", default=".", help="Output directory")
    parser.add_argument(
        "-n",
        "-chunksize",
        type=int,
        default=10_000_000,
        help="Number of lines to read at once, adjust for memory consumption.",
    )
    parser.add_argument(
        "-H", "-Header", action="store_true", help="Flag for if input file has header"
    )
    parser.add_argument(
        "-f", "-field_separator", default="\t", help="Separator for table"
    )
    parser.add_argument("-filter", help="List of IDs to filter for")
    args = parser.parse_args()

    cluster_stream(
        args.e,
        args.c,
        args.q,
        args.t,
        args.s,
        args.l,
        args.d,
        args.n,
        args.H,
        args.f,
        args.filter,
    )
