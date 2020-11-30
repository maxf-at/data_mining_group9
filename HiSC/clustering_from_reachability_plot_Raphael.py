import numpy as np
import matplotlib.pyplot as plt

# finding the threshold
def find_threshold(y, k=6):
    """
    k is the number of classes (>= 2)
    
    note that I count the point that is at the threshold not as 
    noise point but as belonging to the next cluster, since this is the
    one point that started the next cluster (the distances suddenly are
    are low because this point has been added to the cluster ordering)
    """

    # remove the inf value 
    y = y[1:]
    
    # start with highest possible threshold
    thres = np.max(y)
    border = 0
    while np.sum(border) < k:
        # find indices where we are higher-equal than threshold in one position
        # and lower than threshold in the next position (we want to find exactly
        # k-1 such positions as they correspond to borders between clusters)
        higherequal = (y >= thres)
        lower = (y < thres)
        # set first to false, since that is inf and we want to ignore it below
        higherequal[0] = False
        # shift lower and set last one to false to ignore (since that is 
        # introduced from the beginning with roll)
        lower_shifted = np.roll(lower, -1)
        lower_shifted[-1] = False
        border = np.logical_and(higherequal, lower_shifted)
              
        if np.sum(border) >= k-1:
            if np.sum(border) > k-1:
                # simple fix of the problem
                print("!!!!!!!!!!!!!!!!!!!\n Couldn't find k unambigous clusters due to tie - returned for next larger possible k\n!!!!!!!!!!!!!!!!!!!")
                
            # noise is labelled with the highest int
            border_indices = np.append(
                np.append([0], np.where(border)), 
                                            [len(y)])
            
            labels = np.repeat(np.arange(len(border_indices)-1),
                               np.diff(border_indices))
            # set all remaining that are higher than thres to noise
            np.put(labels, np.where(y > thres), np.max(labels)+1)
            
            # for testing #####################################
            plt.axhline(y=thres, color="k")                
            plt.scatter(np.arange(1, len(y)+1), y, marker="x", c=labels)
            plt.title(f"k={k}")
            ###################################################
            
            return labels, thres

        # next step
        thres -= 1
        
        if thres < 0:
            raise UserWarning("Not that many clusters found! Try a lower cluster number")
    
    
# first value is added while still with inf distance
y = np.array([np.inf, 5, 1, 2, 3, 1, 1, 3, 4, 5, 6, 2, 4, 7])
# y = np.array([np.inf, 5, 1, 2, 3, 1, 1, 3, 4, 5, 6, 2, 4, 7])
k = 3

labels, thres = find_threshold(y, k)



