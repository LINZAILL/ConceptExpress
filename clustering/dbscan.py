# pip install scikit-learn matplotlib 
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.sparse as sp
import warnings
from ptp_utils import kld_distance, emd_distance_2d, get_connect, high_dim_connect
import time
import argparse

#compute adj and original distance
def clust_rank(
        mat,
        use_ann_above_samples,
        initial_rank=None,
        distance='cosine',
        verbose=False):

    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = np.empty(shape=(1, 1))
    elif s <= use_ann_above_samples:
        if distance != 'kld' and distance != 'emd':
            orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        elif distance == 'kld':
            orig_dist = kld_distance(mat, mat)
        elif distance == 'emd':
            orig_dist = emd_distance_2d(mat, mat)
            
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(use_ann_above_samples))
        if verbose:
            print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    
    #print("orig_dist:",orig_dist)
    
    return A, orig_dist
     
def compute_min_sim_list(c,orig_dist,num_clust):
    min_sim_list = []

    mat = np.zeros([num_clust,num_clust])
    for i in range(num_clust):
        for j in range(num_clust):
        
            if i == j:
                continue
            
            classed_i_index = (c == i)
            classed_i_index = classed_i_index.reshape(-1,1)
     #       print("classed_i_index",classed_i_index)
            classed_j_index = (c == j)
            classed_j_index = classed_j_index.reshape(1,-1)
      #      print("classed_j_index",classed_j_index)
            selection_ij = classed_i_index*classed_j_index
       #     print("selection_ij",selection_ij)
            class_dist = selection_ij*orig_dist
            mat[i,j] = np.mean(class_dist)
    
    print("mat:",mat)
      
    min_sim_list.append(np.max(mat))
    print("min_sim_list:",min_sim_list)
    
    return min_sim_list




def DBscan(data):
  # Create a DBSCAN instance
  # epsilon = 0.0  # Epsilon radius
  # min_samples = 5  # Minimum number of points to form a core point
  min_samples = 5  # Minimum number of points to form a core point
  
  num_clust_specified = 10
  epsilon_min = 0.01
  epsilon_max = 0.5
  epsilon_mid = (epsilon_min + epsilon_max) / 2
  epsilon = epsilon_mid
  
  dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
  
  # dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric=kld_distance)
  # Fit the DBSCAN model to the data
  labels = dbscan.fit_predict(data)
  # c N*p matrix
  c = labels
  # Number of clusters in labels, ignoring noise points (-1)
  num_clust = len(set(labels)) - (1 if -1 in labels else 0)
  
  while num_clust != num_clust_specified:
      if num_clust < num_clust_specified:
          epsilon_max = epsilon_mid
      elif num_clust > num_clust_specified:
          epsilon_min = epsilon_mid
      epsilon_mid = (epsilon_min + epsilon_max) / 2
      epsilon = epsilon_mid
  
      dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
      # dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric=kld_distance)
      # Fit the DBSCAN model to the data
      labels = dbscan.fit_predict(data)
      # c N*p matrix
      c = labels
      # Number of clusters in labels, ignoring noise points (-1)
      
      num_clust = len(set(labels)) - (1 if -1 in labels else 0)
  print("nb_clust:",num_clust)
  print('epsilon',epsilon)
 # print("nb_clust:",num_clust)
  
  
  
  # req_c
  req_c = None
  #min_sim_list
  use_ann_above_samples = 7000
  adj, orig_dist = clust_rank(data,use_ann_above_samples, initial_rank=None,distance='cosine', verbose=False)
  # adj, orig_dist = clust_rank(data,use_ann_above_samples, initial_rank=None,distance='kld', verbose=False)
  min_sim_list = compute_min_sim_list(c,orig_dist,num_clust)
  
  
  return c, num_clust, req_c, min_sim_list


