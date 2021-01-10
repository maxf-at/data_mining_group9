# Script if you want to generate the kernels yourself
# Important: You do not need to generate your own kernels for Task 1 you can just download them here: 
# https://ucloud.univie.ac.at/index.php/s/E3YKph0jkpbw8TN

#!/usr/bin/env python
# coding: utf-8

# #Download the TUDataset Repository with
# #git clone https://github.com/chrsmrrs/tudataset.git
# #move this script to tudataset/tud_benchmark

# #Install pytorch geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# #Here is the gpu cuda installation, for the cpu version replace cu102 with cpu
# pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip install torch-geometric

# pip --no-cache-dir install pybind11
# sudo apt-get install libeigen3-dev

# compile kernel_baselines package
# g++ -I /usr/include/eigen3 -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`

import os
import numpy as np
import kernel_baselines as kb
from auxiliarymethods import datasets as dp
from scipy.sparse import save_npz

def setup_directory(dir_name, verbose=False):
    """
    Setup directory in case it does not exist
    Parameters:
    -------------
    dir_name: str, path + name to directory
    verbose: bool, indicates whether directory creation should be printed or not.
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            if verbose:
                print("Created Directory: {}".format(dir_name))
        except Exception as e:
            raise RuntimeError(
                "Could not create directory: {}\n {}".format(dir_name, e))


def main():
    use_edge_labels = False
    for USE_LABELS in [True, False]:# Except IMDB-BINARY
        for dataset, use_labels in [["IMDB-BINARY", False],["MSRC_21",USE_LABELS], ["NCI1", USE_LABELS], ["ENZYMES", USE_LABELS]]:
            if use_labels:
                base_path = os.path.join("kernels","node_labels")
            else:
                base_path = os.path.join("kernels","without_labels")
            setup_directory(base_path)
            print("Start processing data set ", dataset)
            # Download dataset.
            classes = dp.get_dataset(dataset)
            # *Weisfeihler-Lehman*
            print("Start computing Weisfeihler-Lehman gram matrix and vector representations")
            iterations = 6
            #0 taking just the nodelabels themselves into account; 
            #1 considers nearest-neighbours, 2 one layer deeper and so on
            for i in range(1, iterations):
                print("Start iteration ", i)
                #Gram Matrix for the Weisfeiler-Lehman subtree kernel
                gram_matrix_wl = kb.compute_wl_1_dense(dataset, i, use_labels, use_edge_labels)
                np.savetxt(os.path.join(base_path,f"{dataset}_gram_matrix_wl{i}.csv"),
                        gram_matrix_wl,
                        delimiter=";")
                #Sparse Vectors for the Weisfeiler-Lehmann subtree kernel
                vectors_wl = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
                save_npz(os.path.join(base_path,f"{dataset}_vectors_wl{i}.npz"),
                        vectors_wl, compressed=True)


            # *Graphlet kernel*
            print("Start computing Graphlet gram matrix")

            #Gram Matrix for the Graphlet kernel
            gram_matrix_graphlet= kb.compute_graphlet_dense(dataset, use_labels, use_edge_labels)
            np.savetxt(os.path.join(base_path,f"{dataset}_gram_matrix_graphlet.csv"),
                    gram_matrix_graphlet,
                    delimiter=";")

            print("Start computing Graphlet vector representation")
            #Sparse Vectors for the Graphlet kernel
            vectors_graphlet = kb.compute_graphlet_sparse(dataset, use_labels, use_edge_labels)
            save_npz(os.path.join(base_path,f"{dataset}_vectors_graphlet.npz"),
                    vectors_graphlet, compressed=True)


            print("Start computing Shortest path gram matrix")

            #Gram Matrix for the Shortest path kernel
            gram_matrix_shortestpath = kb.compute_shortestpath_dense(dataset, use_labels)
            np.savetxt(os.path.join(base_path,f"{dataset}_gram_matrix_shortestpath.csv"),
                    gram_matrix_shortestpath,
                    delimiter=";")

            print("Start computing Shortest path vector representation")

            #Sparse Vectors for the Shortest path kernel
            vectors_shortestpath = kb.compute_shortestpath_sparse(dataset, use_labels)
            save_npz(os.path.join(base_path,f"{dataset}_vectors_shortestpath.npz"),
                    vectors_shortestpath, compressed=True)


if __name__ == "__main__":
    main()

# zip all kernels
# zip kernels/node_labels/MSRC_21_graph_kernels.zip kernels/node_labels/MSRC_21_*.csv
# zip kernels/without_labels/MSRC_21_graph_kernels.zip kernels/without_labels/MSRC_21_*.csv

# zip kernels/node_labels/ENZYMES_graph_kernels.zip kernels/node_labels/ENZYMES_*.csv
# zip kernels/without_labels/ENZYMES_graph_kernels.zip kernels/without_labels/ENZYMES_*.csv

# zip kernels/node_labels/NCI1_graph_kernels.zip kernels/node_labels/NCI1_*.csv
# zip kernels/without_labels/NCI1_graph_kernels.zip kernels/without_labels/NCI1_*.csv

# zip kernels/without_labels/IMDB-BINARY_graph_kernels.zip kernels/without_labels/IMDB-BINARY_*.csv
