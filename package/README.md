# Protein Cluster Tools
By John Chen and Barnabas Gall. Usage examples on [GitHub](https://github.com/johnchen93/ProteinClusterTools).

A pipeline for analyzing protein families as clusters of related sequences.

# Usage
See details on installation below.  
  
This pipeline provides different approaches for grouping (clustering) protein sequences in sequence space, and provides ways to visualize these clusterings in an interactive manner. 

The work flow broadly follows 3 steps:  
1. Starting with a fasta file of protein sequences, create protein clusterings based on homology or vector representation based methods. See the [run pipeline](/analyses/run_pipeline.ipynb) notebook.
2. Once the cluster definitions are made, create interactive visualizations of the sequence space, with user defined annotations. The main methods featured in this pipeline are:  
    
    a. A hierarchical cluster plot that captures cluster separations broadly at different levels of clustering. Works for all methods of clustering in 1). See the [hierarchical cluster plot](/analyses/hierarchical_cluster_plot.ipynb) notebook.  
    b. A tree structure that shows detailed pairwise separations for all sequences (from indivual sequences to fully clustered). This is specifically for the vector representation based hierarchical clustering. See the [tree structure](/analyses/tree_structure.ipynb) notebook.  
    c. Plotting the vector representation of each sequence individually after a UMAP dimensionality reduction. See the [UMAP plot](/analyses/UMAP_plot.ipynb) notebook.
3. When the user has explored sequence space, they can select desired representative sequences from target clusters using either an HMM or vector based approach. See [representative selection](/analyses/representative_selection.ipynb), or end of [tree structure](/analyses/tree_structure.ipynb) in the case of dealing with targetting any node in a tree structure.
        

# Installation
## All-in-one install using conda
**It is recommended to use a package manager like miniconda to install the required environment.**  If the user prefers a more modular install, see **Details on dependencies** below.

**Note on operating system:** The pipeline is in Python, but some dependencies require a Linux environment. If running this on Windows, it is recommended to use Windows Subsystem for Linux (WSL). Alternatively, windows users can SSH into a Linux computer. 

**Note on hardware:** Part of the pipeline requires a GPU for using protein language models and vector representations. It is recommended to install on a computer that has a GPU (in theory CPU will also work but will be slower). Otherwise, the homology based method can be run on just CPU. The env.yaml for installation *assumes the user has an Nvidia GPU (i.e., uses CUDA)*, if using an AMD GPU the user can try installing pytorch manually by pip (see below).

**Note for conda:** When using conda, it is recommended to install Mamba into the base environment for faster dependency resolution. Then use the mamba command in place of the conda command (they are mostly interchangeable).
```
conda install mamba
```

The command to install all requirements using one command is as follows:
```
conda env create -f env.yaml
```
or if using mamba:
```
mamba env create -f env.yaml
```
This will create a new environment called "proteinclustertools". To run code from the command line, be sure to activate this environment first. For Jupyter notebooks (such as the examples in [analyses](analyses/)), select this as the kernel.
```
conda activate proteinclustertools
```

## Details on dependencies
The pipeline makes use of various software. The benefit of installing using conda is that the virtual environment can also handle non-Python software, helping avoid version conflicts (such as for CUDA). In other cases, installing with can be easier with conda (same 'install' command), and avoids needing different details for each separate tool.

### Python dependencies
For most Python packages, if the user does not wish to use conda, they can instead use pip. The following packages are listed in the 'req.txt' file that can be used with pip.
```
pandas
matplotlib
biopython
seaborn
scikit-learn
scipy
fastcluster
Box2d>=2.3.10
tqdm
pyyaml
ipykernel
bokeh
umap-learn
fair-esm
proteinclustertools
```
The following command can then be used to install.
```
pip install -r req.txt
```

### PyTorch
Part of the package requires PyTorch. By itself, PyTorch should still run on CPU (slower and may take quite a bit of RAM, requiring lower tokens per batch). But if the user wishes to leverage their GPU, try installing using the pip commands from the [PyTorch webtool](https://pytorch.org/get-started/locally/). Pay attension to the version, which has to match the GPU CUDA or ROCm driver.
  
Nvidia (CUDA) example:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

AMD (ROCm):
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

The user also has to install CUDA or ROCm separately.

### Other command line tools
There are 3 command line tools used in this package that would need to be installed separately. See the following links for their install instructions.
* [MMSeqs](https://github.com/soedinglab/MMseqs2) - for all-by-all pairwise alignments in the homology based method
* [MAFFT](https://mafft.cbrc.jp/alignment/software/) - for creating multiple sequence alignments during representative selection
* [HMMER](http://hmmer.org/documentation.html) - for creating HMM profiles during representative selection
