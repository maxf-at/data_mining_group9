
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from dataclasses import dataclass, field
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import metrics


"""
HiSC algorithm for VU Data Mining 2020
Currently, it is advised to call the main function from a Jupyter Notebook,
see HiSC_sample_notebook.ipynb
"""


# struct for all datapoints
@dataclass
class Point:
    id:           int
    predecessor:  int
    subspace_dim: int
    distance:     float
    label:        int        
       
    # overload string output
    def __str__(self):
        return f'id: {self.id:5}, pred: {self.predecessor:5}, subs: {self.subspace_dim:2}, dist: {self.distance:.5f}, label: {self.label:5}'

    
# @param data: dataset to order
# @param alpha: threshold value
# @param k: number of neighbors
# @param verbose: set False
# @return cluster_order: returns the new order of the dataset containing points with the given structure
def HiSC(data, alpha, k, verbose=False, elki=True):
    """
    Performs the HiSC algorithm.
    Data has the features in the columns and the samples per rows.
    
    function uses various helper functions (bottom of file)
    
    """
    cluster_order = []
    
    # initialize empty priority queue (list that we will keep sorted with bisect)
    pq = []
    
    # get the nearest neighbours
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(data)
    knn_distances, knn_indices = nbrs.kneighbors(data)
    
    # compute w_p
    wp = subsp_pref_vecs(data, knn_indices, alpha)
    
    if verbose:
        print("Subspace vectors:\n", wp)
    
    # init cluster ordering
    cluster_order = []
    
    # insert the datapoints into the pq, where each point has  
    # id, predecessor, d1, d2 (with respect to the first point).


    for id in range(data.shape[0]):        
        subspace_dim = d1(data, wp, id, 0, alpha)
        distance = d2_elki(data, wp, id, 0)   
        pq.append(Point(id, 0, subspace_dim, distance, -1))



    print ("Running HiSC, input dataset has", len(data), "entries with", len(data[0]), "dimensions")
    
    # outer loop for point o
    while len(pq) != 0:
        if verbose:
            print("PQ:", pq, "\nCluster Order:", cluster_order)
        
        # remove first element per iteration
        o = pq.pop(0) 
                
        # % complete graph, idea from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        percent_complete = (len(data)-len(pq))/len(data)*100       
        bar_length = 20
        filled_length = int(percent_complete/100*bar_length)
        rest = bar_length - filled_length
        bar = "█" * filled_length + '_' * rest
    
        if not verbose:
            print(f'\rComputing hierarchical structure |{bar}| {percent_complete:.1f}% complete', end = "\r")


        ## update the distances
        
        # list that remembers the indices in need of updating, so that we can 
        # go back over the list and remove them to not scramble things up
        needs_updating = []
        
        # inner loop, point p
        for i in range(len(pq)):
            # calculate the distance to the o that was added last
            p = pq[i]
            subspace_dim = d1(data, wp, o.id, p.id, alpha)
            if(elki):
                distance = d2_elki(data, wp, o.id, p.id)
            else:
                distance = d2_paper(data, wp, o.id, p.id)
                        
            # remember the index of p if it needs updating
            # as well as the tuple to update with
            if subspace_dim < p.subspace_dim  or\
               (subspace_dim == p.subspace_dim and distance < p.distance): 
                pq[i] = Point(p.id, o.id, subspace_dim, distance, -1)

                    
                    
        if(elki):
            pq.sort(key=lambda x: (-x.subspace_dim, +x.distance, -x.id))
        else:
            pq.sort(key=lambda x: (x.subspace_dim, +x.distance, -x.id))
        
        # append o to the cluster order
        cluster_order.append(o)
    
    return cluster_order


# @param clus_ord: dataset to cluster
# @return y: returns an array with the labels of the dataset
# @return threshold: number which divides dataset in noise points and clusters
# @min_items: consecutive data points required to count as cluster
def get_clusters(clust_ord, threshold=0.21, min_items=20):
    """
    simple function to assign labels according to a threshold
    to start a new label, at least min_items have to be present
    """
    
    # start new label vector
    y = np.zeros(len(clust_ord), dtype=int)
    i = 0    

    # list of clusters    
    list_of_clusters = []    
    current_list_of_clusters = []
    
    for current_cluster in clust_ord:
        if current_cluster.distance < threshold:
            current_list_of_clusters.append(current_cluster)    
        else:
            list_of_clusters.append(current_list_of_clusters)
            current_list_of_clusters = []
            
    list_of_clusters.append(current_list_of_clusters)    
    
    i = 0
    
    while True:
        current_cluster = list_of_clusters[i]
#         print (len(current_cluster))
        if len(current_cluster) < min_items:
            list_of_clusters.pop(i)
        else:
            i += 1
        if i+1>len(list_of_clusters):
            break   
    for cluster_id, cluster in enumerate(list_of_clusters):        
        for point in cluster:
            y[point.id] = cluster_id+1
    for i in range(len(y)):
        y[i] -= 1 # -1 offset so that noise is -1    
    
    return y # return labels


# @param data: true dataset
# @param cluster_order: ordered dataset
# @param labels_true: labels of true dataset
# @param labels_of_data: labels of ordered dataset
# @param labels: set of the labels_of_data
# @param threshold: number which divides dataset in noise points and clusters
# @return: plots the reachability and some interesting facts about the cluster algorithm
def reachability_plot(data, cluster_order, labels, threshold, predecessor_graph=True, figsize=(16, 8), dimensions=[(0,1),(1,2)]):
    """
    Reachability plot for HiSC
    input: datapoints 
           cluster ordering
           labels (either predicted or true labels)
           threshold (for reachability plot)
    
    inspired by: https://github.com/SamaelChen/hexo-practice-code/blob/master/fun/OPTICS.ipynb    
    """

    # helper function
    # 2d plot from different input dimensions
    def sub_plot(dim1, dim2):
        # compute predecessor lines
        line_plot = []    
        if predecessor_graph:
            for i, point in enumerate(cluster_order):             
                if point.distance <= 0.0:
                    continue               
                
                # we can exclude noise points from the predecessor plots if required...
#                 if cluster_order[point.predecessor].label == -1:
#                     continue                   
                point_xdim = data[point.id][dim1]
                point_ydim = data[point.id][dim2]        
                point_pred_xdim = data[point.predecessor][dim1]
                point_pred_ydim = data[point.predecessor][dim2]        
                line_plot.append((point_xdim, point_ydim, point_pred_xdim, point_pred_ydim))

        # Get the current Axes instance 
        ax = plt.gca()
        for x1, y1, x2, y2 in line_plot:
            # plot lines between 2 points
            line = lines.Line2D([x1, x2], [y1, y2], lw=1, color='gray', axes=ax)
            ax.add_line(line)

        unique_labels = set(labels) 
        colors = [plt.cm.Spectral(color) for color in np.linspace(0, 1, len(unique_labels))]
        
        for label, color in zip(unique_labels, colors):
            # set noise as black if -1 label available
            if label == -1 or label == "-1" or label == "Noise" or label == "noise": 
                color = (0, 0, 0, 1)
            class_member_mask = (labels == label)
            xy = data[class_member_mask]        
            plt.plot(xy[:, dim1], xy[:, dim2], 'o', markerfacecolor=tuple(color), markeredgewidth=0.5, markeredgecolor='k', markersize=5)

        plt.title(f'x-axis: dimension {dim1}; y-axis: dimension {dim2}')
#         plt.legend(unique_labels)
        plt.grid(True, alpha=0.3)

    
    # set point labels according to input labels
    for i, point in enumerate(cluster_order):
        point.label = labels[i]
   
    # how many subplots? 
    y_plots = int(np.ceil(len(dimensions)/2))
    
    # always plot 2 plots next to each other
    for y_plot in range(y_plots):       
       # setup size
        plt.figure(figsize=figsize)           
        #  left top plot: first dimensions
        dim1 = dimensions[y_plot*2][0]
        dim2 = dimensions[y_plot*2][1]
        plt.subplot(1, 2, 1)
        sub_plot(dim1, dim2)    
        # right top plot: other dimensions
        if y_plot*2+1>=len(dimensions):break
        dim1 = dimensions[y_plot*2+1][0]
        dim2 = dimensions[y_plot*2+1][1]
        plt.subplot(1, 2, 2)        
        sub_plot(dim1, dim2) 
    
            
    # plot reachability bargraph
    plt.figure(figsize=figsize)  
    unique_labels = set(labels)    
    # using a matplotlib colourmap
    colors = [plt.cm.Spectral(color) for color in np.linspace(0, 1, len(unique_labels))]
    
    bar_x = []
    bar_y = []    
    for label, color in zip(unique_labels, colors):        
        label = int(label)     
        # set noise as black if -1 label available
        if label == -1 or label == "-1" or label == "Noise" or label == "noise": 
            color = (0, 0, 0, 1)
        for i, point in enumerate(cluster_order):                 
            if int(labels[point.id]) == label:
                bar_x.append(i)
                bar_y.append(point.distance)                
                
        plt.bar(bar_x, bar_y, color=tuple(color), label=label)
        # reset colors for next label set
        bar_x = []
        bar_y = []
        
    # plot threshold line
    plt.plot(np.arange(len(data)), [threshold] * len(data), '--', color='red')
        
    ax = plt.gca()
    plt.title('Reachability Plot')   
    plt.grid(False)      
    plt.xticks([]) # remove x axis labels - too much information
    plt.xlabel('datapoints')
    plt.ylabel('subspace weighted distance')
    plt.legend(prop={'size': 16})
    plt.show()
    


####################################
# helper functions for HiSC below: #
####################################

def process_csv(input_filename, sep=" ", input_filename_labels="", true_labels=True):
    """
    simple function to read csv files, save X and y
    can be adapted to use MinMax preprocessing to scale each dimension from 0 to 1
    """    
    # take labels from extra file
    if input_filename_labels!="":
        data = np.genfromtxt(input_filename, delimiter=sep,dtype='str')
        X = data.astype(float) 
        labels = np.genfromtxt(input_filename_labels, delimiter=sep,dtype='str')
        y = labels.astype(int) 
        return X, y
    # take labels from the last column
    elif true_labels: 
        data = np.genfromtxt(input_filename, delimiter=sep,dtype='str')
        X = data[:,:-1].astype(float) # exclude last column --> labels
        y = data[:,-1] # labels
        return X, y
    else:
        data = np.genfromtxt(input_filename, delimiter=sep,dtype='str')
        X = data.astype(float) 
        return X
    
    # we may or may not need this later...
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     X = min_max_scaler.fit_transform(X)    
    

def subsp_pref_vecs(data, knn_indices, alpha):
    """
    Input:
        data, np array of shape (n, dimensions), where each row is a data point.
        knn_indices, np array of shape (n, k), where each row is a nearest 
        neighbour pairing and the first instance is the central node. 
        alpha, float, threshold for high variance.
    Returns:
        a matrix containing all the subspace preference vectors,
        as a numpy array.
    """
    k = knn_indices.shape[1]
    
    ## variance 
    q = data[knn_indices]
    # create p to subtract it from values of q in the right pattern
    p = np.repeat(data[knn_indices][:,0,:], repeats=k, axis=0).reshape(q.shape)
    var = np.sum((p-q)**2, axis=1) / k
    
    # return subspace preference vectors
    return var <= alpha


def d1(data, wp, p, q, alpha):
    """
    Input:
        wp (np.array) - a matrix containing all the subspace preference 
        vectors in its rows
        p, q (int) - indices indicating which wp's to compare.
        data, np array of shape (n, dimensions), where each row is a data 
        point. 
        alpha (int), threshold for high variance
    Output:
        d1 (int) as referred to in defintion 4
    """
    if distw_pq(data, wp, p, q) > alpha or distw_pq(data, wp, q, p) > alpha:
        return lambda_pq(wp, p, q) + 1
    else:
        return lambda_pq(wp, p, q)

    
def d2_elki(data, wp, p, q):
    """
    Input:
        wp (np.array) - a matrix containing all the subspace preference 
        vectors in its rows
        p, q (int) - indices indicating which wp's to compare.
        data, np array of shape (n, dimensions), where each row is a data 
        point. 
    Output:
        d2 (int) as defined in defition 4
    
    probably different than in the paper, see below
    """
    wp_inv = np.logical_and(wp[p], wp[q])

    # return euclidian distance weighted by inverse combined pref vec
    return (np.sum( (data[p] - data[q])**2 * wp_inv ))**(1/2)


def d2_paper(data, wp, p, q):
    """
    Input:
        wp (np.array) - a matrix containing all the subspace preference 
        vectors in its rows
        p, q (int) - indices indicating which wp's to compare.
        data, np array of shape (n, dimensions), where each row is a data 
        point. 
    Output:
        d2 (int) as defined in defition 4
    
    again sth different than in the paper, see below
    """
    # inverse combined preference vector
    wp_inv = np.logical_not( np.logical_and( wp[p], wp[q]))

    # return euclidian distance weighted by inverse combined pref vec
    return (np.sum( (data[p] - data[q])**2 * wp_inv ))**(1/2)


def lambda_pq(wp, p, q):
    """
    Input:
        wp (np.array) - a matrix containing all the subspace preference 
        vectors in its rows
        p, q (int) - indices indicating which wp's to compare.
    Output:
        subspace dimensionality (int), denoted as lamda(p, q) in the 
        definition 3 in the paper.
    """
    return np.sum( np.logical_not( np.logical_and( wp[p], wp[q])))


def distw_pq(data, wp, p, q):
    """
    Input: 
        wp (np.array) - a matrix containing all the subspace preference 
        vectors in its rows
        p, q (int) - indices indicating which wp's to compare. 
        data, np array of shape (n, dimensions), where each row is a data 
        point.
    Output:
        Weighted euclidian distance between the vectors with index p and q
        as given in definition 4.
    
    In definition 4 the square root at the end is missing in comparison to 
    the paper.
    """
    return (np.sum( (data[p] - data[q])**2 * wp[p] ))**(1/2)