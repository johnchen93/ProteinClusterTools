import pickle
import os
from multiprocessing import Pool
import random

def merge_overlapping_sets(list1, list2):
    """
    Input requires the sets in each list to be disjoint amongst each othor, but not between lists.
    """
    merged_sets = []  # To keep track of merged sets for later addition
    to_remove = set()  # To keep track of indices in list2 that have been merged

    # filter out singles
    singles = [s for s in list1 if len(s) == 1]
    singles.extend([s for s in list2 if len(s) == 1])
    list1 = [s for s in list1 if len(s) > 1]
    list2 = [s for s in list2 if len(s) > 1]

    disjoint=[]
    # Iterate over each set in the first list
    for set1 in list1:
        # Iterate over each set in the second list by index and value
        overlapping=[]
        for i, set2 in enumerate(list2):
            if i in to_remove:
                # Skip sets that are already merged
                continue
            # Check for intersection
            if not set1.isdisjoint(set2):
                overlapping.append(i)

        # Add the merged set to the list of merged sets
        if len(overlapping) > 0:
            list2[overlapping[0]].update(set1, *[list2[i] for i in overlapping[1:]])
            for i in overlapping[1:]:
                list2[i]=None
                to_remove.add(i)  # Mark this set as merged
        else:
            disjoint.append(set1)

    # Rebuild list2 without the merged sets
    list2[:] = [s for i, s in enumerate(list2) if i not in to_remove]
    list2.extend(disjoint)

    grouped=set()
    for i, s in enumerate(singles):
        (item,)=s
        for set2 in list2:
            if item in set2:
                grouped.add(i)
                break
    singles = [s for i, s in enumerate(singles) if i not in grouped]
    list2.extend(singles)
    
    return list2

def merge_pair(pair):
    """
    Wrapper function for merging a pair of datasets.
    """
    data=[]
    for item in pair:
        if type(item) is str:
            # load the file and take the dict values
            with open(item, 'rb') as f:
                item = list(pickle.load(f).values())
        data.append(item)

    return merge_overlapping_sets(data[0], data[1])

def pairwise_merge(datasets, max_datasets=100):
    """
    Function to merge datasets pairwise using multiprocessing.
    """
    if len(datasets) < 2:
        return datasets
    # shuffle the datasets
    random.shuffle(datasets)

    # If we have more datasets than the maximum, hold out the extras
    hold_out=[]
    if len(datasets) > max_datasets:
        hold_out = datasets[max_datasets:]
        datasets = datasets[:max_datasets]

    # Ensure we have an even number of datasets for pairing; if not, strip the last element
    left_over = None
    if len(datasets) % 2 != 0:
        left_over = datasets.pop()

    # Pair up datasets for merging
    pairs = zip(*[iter(datasets)]*2)

    # Use multiprocessing Pool to merge pairs in parallel
    with Pool() as pool:
        merged_datasets = list(pool.imap_unordered(merge_pair, pairs)) # using list forces blocking

    if hold_out:
        merged_datasets.extend(hold_out)
    if left_over:
        # If we stripped a dataset, add it back to the merged datasets
        merged_datasets.append(left_over)

    return merged_datasets

import time
def iterative_merge(datasets, max_datasets=None):
    """
    Repeatedly apply pairwise merging until one dataset remains.
    """
    # in the case that merging is not needed
    if len(datasets) == 1:
        # load the dataset with pickle
        with open(datasets[0], 'rb') as f:
            data=list(pickle.load(f).values())
        return data
    
    # Start the merging process
    cycle=1
    while len(datasets) > 1:
        if max_datasets is None:
            max_datasets = len(datasets)
        print('Cycle:',cycle, f'  Datasets: {min(len(datasets), max_datasets)}/{len(datasets)}')
        start=time.time()
        datasets = pairwise_merge(datasets, max_datasets) # actual function
        print(f' - completed in {time.time()-start:.2f} s')
        cycle+=1
    return datasets[0]

import pandas as pd
def SetsToDataFrame(dataset):
    """
    Convert a list of sets to a DataFrame.
    """
    
    # sort by size
    dataset = sorted(dataset, key=len, reverse=True)

    data={}
    for i, s in enumerate(dataset):
        for x in s:
            data[x]=i
    return pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index':'id', 0:'cluster'})