# TODO © Namen
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import metrics

np.set_printoptions(linewidth=120, formatter={'float': '{: 0.3f}'.format})

# struct for all datapoints like elki
@dataclass
class Point_elki:
    id:           int
    predecessor:  int
    subspace_dim: int
    distance:     float
    label:        int
        
    # overload < (less than) operator    
    def __lt__(self, point_2):
        return (-self.subspace_dim, +self.distance, -self.id) < (-point_2.subspace_dim, +point_2.distance, -point_2.id)           
    # overload string output
    def __str__(self):
        return f'id: {self.id:5}, pred: {self.predecessor:5}, subs: {self.subspace_dim:2}, dist: {self.distance:.5f}, label: {self.label:5}'

# struct for all datapoints like paper
@dataclass
class Point_paper:
    id:           int
    predecessor:  int
    subspace_dim: int
    distance:     float
    label:        int
        
    # overload < (less than) operator    
    def __lt__(self, point_2):
        return (self.subspace_dim, +self.distance, -self.id) < (point_2.subspace_dim, +point_2.distance, -point_2.id)           
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
    
    # the first item is chosen first by default and will be ignored for the pq
    # cluster_order.append((np.inf, np.inf, 0))
    cluster_order = []
    
    # insert the remaining datapoints into the pq, where each is a tuple of  
    # d1, d2 (with respect to the first point) and its index in the dataset. 
    # This list will be kept in ascending order, even though popping from the end
    # is faster, because bisect only works that way
    if(elki):
        for id in range(data.shape[0]):        
            subspace_dim = d1(data, wp, id, 0, alpha)
            distance = d2_elki(data, wp, id, 0)   
            pq.append(Point_elki(id, 0, subspace_dim, distance, -1))
    else:
        for id in range(data.shape[0]):        
            subspace_dim = d1(data, wp, id, 0, alpha)
            distance = d2_paper(data, wp, id, 0)   
            pq.append(Point_paper(id, 0, subspace_dim, distance, -1))

    # TODO 
    # sort everything to get a priority queue that we will keep sorted
    # pq.sort()
    # not necessary, initially all are sorted by id anyway
    
    while len(pq) != 0:
        if verbose:
            print("\n######\nPQ:", pq, "\n\nCluster Order:", cluster_order)
        
        o = pq.pop(0) 
        # TODO
        # !!!!!!!!!!!!!!!
        # for basic_ext it seems to be better to pop from the end, i.e. 
        # take the one with the largest distance (which does not make sense)
        # at least that corresponds to the ELKI debugging we did
        # but it messes up the basic example
        
        if verbose:
            print("o:", o)
        
        ## update the distances
        
        # list that remembers the indices in need of updating, so that we can 
        # go back over the list and remove them to not scramble things up
        needs_updating = []
        
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
                if(elki):
                   pq[i] = Point_elki(p.id, o.id, subspace_dim, distance, -1)
                else:
                    pq[i] = Point_paper(p.id, o.id, subspace_dim, distance, -1)
                    
        pq.sort(key=lambda x: (-x.subspace_dim, +x.distance, -x.id))
        
        # append o to the cluster order
        cluster_order.append(o)
    
    return cluster_order

# @param cluster_order: dataset to cluster
# @param n_cluster: number of clusters
# @return cluster_order: returns the dataset with new labels (clusters)
# @return labels_of_data: returns an array with the labels of the dataset
# @return labels: returns a set of the labels_of_data to show which clusters are assigned
# @return threshold: number which divides dataset in noise points and clusters
def get_clusters(cluster_order, n_cluster):
    threshold = get_threshold(cluster_order, n_cluster)
    labels_of_data = np.zeros(len(cluster_order), dtype=int)
    cluster = 1
    first = True
    k = 0
    i = 0
  
    for i in range(len(cluster_order)):
        if(cluster_order[i].subspace_dim < threshold):  
            if(first):
                cluster_order[i].label = cluster
                first = False
            elif(cluster_order[i].subspace_dim == cluster_order[i-1].subspace_dim):
                cluster_order[i].label = cluster_order[i-1].label
            elif(cluster_order[i].subspace_dim != cluster_order[i-1].subspace_dim):
                cluster += 1
                cluster_order[i].label = cluster

    for p in cluster_order:
        labels_of_data[k] = p.label
        k += 1
    
    labels = set(labels_of_data)

    return cluster_order, labels_of_data, labels, threshold

# @param data: true dataset
# @param cluster_order: ordered dataset
# @param labels_true: labels of true dataset
# @param labels_of_data: labels of ordered dataset
# @param labels: set of the labels_of_data
# @param threshold: number which divides dataset in noise points and clusters
# @return: plots the reachability and some interesting facts about the cluster algorithm
def reachability_plot(data, cluster_order, labels_true, labels_of_data, labels, threshold):
    n_clusters_ = len(labels) - (1 if -1 else 0)
    n_noise_ = list(labels_of_data).count(-1)
    sequence = []
    subspaceDim = []

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_of_data))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_of_data))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_of_data))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels_of_data))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels_of_data))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels_of_data))

    #plot reachability
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    unique_labels = set(labels_true)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == 'Noise':
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels_true == k)

        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8)
    ax = plt.gca()
    ax.set_facecolor('gray')
    plt.title('Ground Truth %d' % len(unique_labels))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(labels))]
    for k, col in zip(labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels_of_data == k)

        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8)
    ax = plt.gca()
    ax.set_facecolor('gray')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.grid(True)

    plt.figure(figsize=(16, 7))
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(labels))]
    for label, color in zip(labels, colors):
        if(label == -1):
            # Black used for noise.
            color = [0, 0, 0, 1]

        for point in cluster_order:
            if(point.label == label):
                sequence.append(str(point.id))
                subspaceDim.append(point.subspace_dim)
        
        if(label == -1):
            plt.plot(sequence, subspaceDim, 'o', markerfacecolor=tuple(color), markersize=8, label = "Noise Point")
        else:
            plt.plot(sequence, subspaceDim, 'o', markerfacecolor=tuple(color), markersize=8, label = "Cluster " + str(label))#

        sequence = []
        subspaceDim = []

    ax = plt.gca()
    ax.set_facecolor('gray')
    plt.title('Reachability')   
    plt.grid(True)  
    plt.plot(np.arange(len(data)), [threshold] * len(data), '--', color='red')
    plt.xlabel('id')
    plt.ylabel('subspace dimensionality')
    plt.legend()

    plt.show()

##################################################

#TODO
def process_csv(input_filename):
    """
    simple function to read csv files, save X and y
    can be adapted to use MinMax preprocessing to scale each dimension from 0 to 1
    """
    data = np.genfromtxt(input_filename, delimiter=' ',dtype='str')
    X = data[:,:-1].astype(float) # exclude last column --> labels
    y = data[:,-1] # labels
    
    # we may or may not need this later...
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     X = min_max_scaler.fit_transform(X)    
    return X, y

#TODO
# preprocessing: calculate preference vectors for all data points
def _distance(vec):
    return np.linalg.norm(vec, 2) # l2 by default

#TODO
def var_a(dimension, p, q):    
    pi_d_p = X[p][dimension]
    pi_d_q = X[q][dimension]    
    return (pi_d_q - pi_d_p)**2

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
    
    !!! again sth different than in the paper, see below
    """
    wp_inv = np.logical_and( wp[p], wp[q])

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
    
    !!! again sth different than in the paper, see below
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
    # TODO
    # not, because we want to count the zeros
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
    
    !! In definition 4 the square root at the end is missing in comparison to 
    !! the paper!
    """
    return (np.sum( (data[p] - data[q])**2 * wp[p] ))**(1/2)

def get_threshold(cluster_order, n_cluster):
    subspace_dim_of_data = np.zeros(len(cluster_order), dtype=int)
    list_for_treshold = []
    k = 0

    for point in cluster_order:
        subspace_dim_of_data[k] = point.subspace_dim
        k += 1

    subspace_dims = set(subspace_dim_of_data)

    for i in range(len(subspace_dims)):
        if(i < n_cluster):
            element = list(subspace_dims)[i]
            list_for_treshold.append(element)

    threshold = list_for_treshold[n_cluster-1] + 1
    return threshold
