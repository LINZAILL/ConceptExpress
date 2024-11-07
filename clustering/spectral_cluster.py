from .dbscan import clust_rank, compute_min_sim_list
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

#def spectral_cluster(X, n_clusters=3, sigma=1, k=5):
#    def graph_building_KNN(X, k=5, sigma=1):
#        N = len(X)
#        S = np.zeros((N, N))
#        print(len(X))
#        for i, x in enumerate(X):
#            if i % 100 == 0:
#                print(i)
#            S[i] = np.array([np.linalg.norm(x - xi) for xi in X])
#            S[i][i] = 0
        
#        graph = np.zeros((N, N))
#        for i, x in enumerate(X):
#            distance_top_n = np.argsort(S[i])[1: k+1]
#            for nid in distance_top_n:
#                graph[i][nid] = np.exp(-S[i][nid] / (2 * sigma ** 2))
#        return graph
    
#    graph = graph_building_KNN(X, k)
    
#    def laplacianMatrix(A):
#        dm = np.sum(A, axis=1)
#        D = np.diag(dm)
#        L = D - A
#        sqrtD = np.diag(1.0 / (dm ** 0.5))
#        return np.dot(np.dot(sqrtD, L), sqrtD)
    
#    L = laplacianMatrix(graph)
    
#    def smallNeigen(L, n_eigen):
#        eigval, eigvec = np.linalg.eig(L)
#        index = list(map(lambda x: x[1], sorted(zip(eigval, range(len(eigval))))[1:n_eigen]))
#        return eigvec[:, index]
#    H = smallNeigen(L, k)
    
#    kmeans = KMeans(n_clusters=n_clusters).fit(H)
#    return kmeans.labels_

def spectral_cluster(X, n_clusters=3, sigma=1, k=5):
    
    print(n_clusters)
    # 计算相似度矩阵，这里使用公式(8)
    def w(x, y, sig=sigma):
        return np.exp(-1.0 * (x - y).T @ (x - y) /(2 * sig**2)) # 高斯径向基核函数

    # 按照高斯径向基核函数计算样本之间的距离
    W = pairwise_distances(X, metric=w)
    
    # 计算度矩阵
    D = np.diag(W.sum(axis=1))
    
    # 计算拉普拉斯矩阵
    L = D - W
    # 拉普拉斯矩阵规范化，不计算也行
    L_norm = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))
    
    # 特征分解，np.linalg.eig()默认按照特征值升序排序了。
    eigenvals, eigvector = np.linalg.eig(L_norm)
    
    # 如果没有升序排序，可以这样做
    # 将特征值按照升序排列
    ind = np.argsort(eigenvals)
    eig_vec_sorted = eigvector[:,ind] #对应的特征向量也要相应调整顺序
    
    # 取出前k个最小的特征值对应的特征向量，注意这里的k和要聚类的簇数一样
    Q = eig_vec_sorted[:, :k] 
    
    # 对新构成的Q矩阵进行聚类
    km = KMeans(n_clusters=n_clusters)
    Q_abs = np.abs(Q)  
    
    # 对Q_abs的行向量聚类，并计算出每个样本所属的类
    y_pred = km.fit_predict(Q_abs)
    print(y_pred)
    print(type(y_pred))
    print(len(y_pred))
    print(set(y_pred))
    
    return y_pred
#    # 根据预测的标签画出所有的样本
#    plt.scatter(X[:,0], X[:,1], c=y_pred)


def SpectralCluster(data):
  # Create a spectral cluster instance
  
  num_clust = 10
  sigma_ = 1
  k_ = 10
  
  labels = spectral_cluster(data, n_clusters=num_clust, sigma=sigma_, k=k_)

  # c N*1 vector
  c = labels
  # Number of clusters in labels, ignoring noise points (-1)
  num_clust = len(set(labels)) - (1 if -1 in labels else 0)

  print("nb_clust:",num_clust)
 # print('sigma',sigma_)
  
  # req_c
  req_c = None
  #min_sim_list
  
  use_ann_above_samples = 7000
  adj, orig_dist = clust_rank(data,use_ann_above_samples, initial_rank=None,distance='cosine', verbose=False)
  min_sim_list = compute_min_sim_list(c,orig_dist,num_clust)
#  min_sim_list = [100]
  
  
  return c, num_clust, req_c, min_sim_list

    
# from sklearn import datasets
# iris = datasets.load_iris()
# from sklearn.decomposition import PCA
# X_reduced = PCA(n_components=2).fit_transform(iris.data)

# y = iris.target
# import matplotlib.pyplot as plt
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)


