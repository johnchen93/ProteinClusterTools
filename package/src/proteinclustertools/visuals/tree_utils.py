from multiprocessing import Pool
import os
from Bio import Phylo
import pandas as pd
from tqdm import tqdm

# list of functions to make
# break tree into pruned tree and subtrees
def DivideTree(treefile, clusterfile):
    '''
    Take a tree and condense based on clustering definitions, into a main tree with just the clusters as leaves
    and all the different subtrees that are children of the main tree.

    Args:
    - treefile (str): path to tree file
    - clusters (str): path to cluster file
    '''

    # read in tree
    tree = Phylo.read(treefile, 'newick')
    clusters=pd.read_csv(clusterfile, dtype=str)

    # get all internal nodes that represent clusters
    anc_data=[data['id'].to_list() for cluster, data in clusters.groupby('cluster') if len(data)>1]
    cluster_order=[cluster for cluster, data in clusters.groupby('cluster') if len(data)>1]
    # do multi-processing to find common_ancestor
    with Pool() as pool:
        results=pool.map(tree.common_ancestor, anc_data)
    anc_nodes=dict(zip(cluster_order, results))

    # split all the subtrees
    subtrees={}
    for cluster, anc_node in anc_nodes.items():
        subtrees[cluster]=Phylo.BaseTree.Tree(root=anc_node)
        # make root node distance 0
        subtrees[cluster].root.branch_length=0
    
    # create the main tree where nodes are collapsed into leaves if they represent clusters
    target_clades=[clade.name for clade in anc_nodes.values()]#+leaves

    PruneTree(tree, target_clades)

    return tree, subtrees, anc_nodes

# Function to prune the tree
def PruneTree(tree, target_clades):
    """
    Prune the tree by removing branches and making internal nodes as leaves.

    Parameters:
    - tree (Phylo.BaseTree.Tree): The tree to prune.
    - target_clades (list of str): List of internal clade names to retain
    """
    # get depths 
    depths = tree.depths()

    # Then, convert internal nodes to leaves if they are in the target_clades
    for clade in tree.get_nonterminals(order="postorder"):
        if clade.name in target_clades:
            # get terminals to determine deepest leaf
            leaf_depths=[depths[leaf] for leaf in clade.get_terminals()]
            depth_diff=max(leaf_depths)-depths[clade]
            clade.branch_length+=depth_diff
            # remove children
            clade.clades = []

# functions to generate ITOL files
def ITOLMultiCategoryHeader(dataset_label, field_colors, field_labels, datatype='piechart'):
    header = (
        f"DATASET_{datatype.upper()}\n"
        # "#In pie chart datasets, each ID is associated to multiple numeric values...\n"
        "SEPARATOR COMMA\n"
        f"DATASET_LABEL,{dataset_label}\n"
        # "COLOR,#ff0000\n"
        f"FIELD_COLORS,{','.join(field_colors)}\n"
        f"FIELD_LABELS,{','.join(field_labels)}\n"
        "#=================================================================#\n"
        "DATA\n"
    )
    return header

def ITOLProportionDataset(annot_table, color_annot, other_color='#bdbdbd', datatype='multibar', percentage=False, outdir=''):
    '''
    Make a dataset file for ITOL to display proportions of categorical data.

    Args:
    - annot_table (pd.DataFrame): table with columns 'id', 'value', 'count'. Can be obtained from 'AnnotateClusters' function (select a level by key).
    - color_annot (dict): dictionary with keys 'categories', 'value', 'method'. Can be obtained from 'ColorAnnot' function.
    - other_color (str): color to use for the 'Other' category. Aggregates all data not in the color_annot
    - datatype (str): type of dataset to make. Either 'multibar' or 'piechart'
    - percentage (bool): whether to display the data as percentage
    - outdir (str): directory to save the file
    '''
    if datatype not in ['multibar', 'piechart']:
        raise ValueError(f"datatype must be either 'multibar' or 'piechart', not {datatype}")
    
    # go through the data set and 
    color_dict=color_annot['categories']
    label = f'{color_annot['value']}_{color_annot['method']}'.capitalize()
    main_colors, main_labels = list(color_dict.values()), list(color_dict.keys())
    field_colors, field_labels = main_colors + [other_color], main_labels + ['Other']
    # more dataset specific 
    extra_fields = '-1,10,' if datatype == 'piechart' else ''
    suffix = '_norm' if percentage else ''

    # iterate through data and create annotation
    with open(os.path.join(outdir, f"{label}_{datatype}_dataset{suffix}.txt"), "w") as file:
        header = ITOLMultiCategoryHeader(f"{label} {datatype}", field_colors, field_labels, datatype)
        file.write(header)
        for id, data in annot_table.groupby('id'):
            # aggregate all data not in field colors as other
            main_colored=data[data['value'].isin(main_labels)]
            other_count=data[~data['value'].isin(main_labels)]['count'].sum()
            data_dict=main_colored.set_index('value')['count'].to_dict()
            data_dict['Other']=other_count
            counts=[data_dict.get(label, 0) for label in field_labels]
            if percentage:
                total=sum(counts)
                counts=[count/total*100 for count in counts]
            data_line = f"{id},{extra_fields}{','.join(map(str, counts))}\n"
            file.write(data_line)

def ITOLColoredRangesDataset(dataset_label, branch_values, color_dict, outdir='', color_other=True, other_color='#bdbdbd'):
    '''
    Make a colored branches dataset file for ITOL

    Args:
    - dataset_label (str): label for the dataset
    - branch_values (dict): dictionary with branch id as key and value as value
    - color_dict (dict): dictionary with value as key and color as value. Obtained from 'ColorAnnot' function (key='categories')
    - outdir (str): directory to save the file
    - color_other (bool): whether to color branches not in the color_dict as 'Other'
    - other_color (str): color to use for the 'Other' category. Aggregates all data not in the color_annot
    '''
    legend_dict={value: color for value, color in color_dict.items()}
    if color_other:
        legend_dict['Other']=other_color
    with open(os.path.join(outdir, f"{dataset_label}_colored_ranges_dataset.txt"), "w") as file:
        header = (
            f"DATASET_RANGE\n"
            "SEPARATOR COMMA\n"
            f"DATASET_LABEL,{dataset_label}\n"
            "COVER_DATASETS,1\n"
            f"LEGEND_COLORS,{','.join(legend_dict.values())}\n"
            f"LEGEND_LABELS,{','.join(legend_dict.keys())}\n"
            f"LEGEND_TITLE,{dataset_label}\n"
            f"LEGEND_SHAPES,{','.join(['1']*len(legend_dict))}\n"
            "#=================================================================#\n"
            "DATA\n"
        )
        file.write(header)
        for branch, value in branch_values.items():
            color=color_dict.get(value, other_color if color_other else None)
            if color is None:
                continue
            # data_line = f"{branch},label,clade,#000000,1,normal,{color}\n"
            data_line=f'{branch},{branch},{color},,,,,,,,,\n'
            file.write(data_line)


def ITOLBinaryDataset(dataset_label, leaf_ids, shape=2, color='#ff0000', outdir='', internal=False):
    '''
    Make a binary dataset file for ITOL

    Args:
    - dataset_label (str): label for the dataset
    - leaf_ids (list of str): list of leaf ids in the tree to highlight
    - shape (int): shape to show in ITOL, 
        -    #1: square
        -    #2: circle
        -    #3: star
        -    #4: right pointing triangle
        -    #5: left pointing triangle
        -    #6: checkmark
    - color (str): color of the markers
    - outdir (str): directory to save the file
    - internal (bool): whether to make an internal node dataset
    '''
    datatype='binary' if not internal else 'symbol'
    outlabel='leaf' if not internal else 'internal'
    with open(os.path.join(outdir, f"{dataset_label}_highlight_{outlabel}_dataset.txt"), "w") as file:
        header_front = (
            f"DATASET_{datatype.upper()}\n"
            "SEPARATOR COMMA\n"
            f"DATASET_LABEL,{dataset_label}\n")
        header_middle = (
            f"COLOR,{color}\n"
            f"FIELD_SHAPES,{shape}\n"
            f"FIELD_LABELS,{dataset_label}\n") if not internal else ("MAXIMUM_SIZE,5\n")
        header_end = (
            "#=================================================================#\n"
            "DATA\n"
        )
        header=header_front+header_middle+header_end
        file.write(header)

        for id in leaf_ids:
            if not internal:
                data_line = f"{id},1\n"
            else:
                data_line = f"{id},{shape},1,{color},1,1\n"
            file.write(data_line)

def ITOLLabelDataset(dataset_label, label_dict, outdir):
    '''
    Make a label dataset file for ITOL

    Args:
    - dataset_label (str): label for the dataset
    - label_dict (dict): dictionary with id as key and label as value
    - outdir (str): directory to save the file
    '''
    with open(os.path.join(outdir, f"{dataset_label}_label_dataset.txt"), "w") as file:
        header = (
            f"DATASET_TEXT\n"
            "SEPARATOR COMMA\n"
            f"DATASET_LABEL,{dataset_label}\n"
            # "COLOR,#000000\n"
             "#=================================================================#\n"
            "DATA\n"
        )
        file.write(header)
        for id, label in label_dict.items():
            data_line = f"{id},{label},-1,#000000,normal,1,0\n"
            file.write(data_line)
        
    