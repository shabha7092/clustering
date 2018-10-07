'''
Kmeans Clustering Algorithm
'''
import os;
import glob;
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import unittest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import axes3d

matplotlib.interactive(True)
dst = scipy.spatial.distance.euclidean

def read_data_to_dataframe(path):
    '''return dataframe from csvfile's'''
    files = glob.glob(os.path.join(path, "*.csv"))
    data = pd.concat((pd.read_csv(file, index_col=None, header=0) for file in files))
    data = data[data['Ads Returned'] !=0]
    data['Impression Fill Rate'] = data['Impressions']/data['Ads Returned']
    data = data[['Impression Fill Rate','Publisher eCPM','CTR']]
    data = data.reset_index(drop=True)
    return data

def transform_data(data):
    data = data.apply(np.cbrt)  
    return data  

def visulaize_data(data):
    data.boxplot()
    plt.show()
    data.hist()
    plt.show()
    return

def scatter_plot(data):
 #  plt.xlim(-5, 15)
 #  plt.ylim(-0.5, 0.8)
    xlabel = data.columns[0]
    ylabel = data.columns[1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(data[xlabel], data[ylabel], marker='+', c = 'red')
    plt.show()
    return

def scatter_plot_label(data):
    plt.xlim(-0.75, 1)
    plt.ylim(-0.4, 0.9)
    xlabel = data.columns[0]
    ylabel = data.columns[1]
    label = data.columns[data.columns.size-1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    colors = ['green','blue','brown']
    plt.scatter(data[xlabel], data[ylabel], marker='+', c=data[label], cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()
    return

def scatter_plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = fig.gca(projection='3d')
    xlabel = data.columns[0]
    ylabel = data.columns[1]
    zlabel = data.columns[2]
    label = data.columns[data.columns.size-1]
    colors = ("red", "green", "blue")
    ax.scatter(data[xlabel], data[ylabel], data[zlabel], marker='+', c=data[label], cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_zlabel(data.columns[2])
    plt.title('Publisher Clustering')
    plt.legend(loc=2)
    plt.show()
    return


def std_scaler(data):
    scaled_features = StandardScaler().fit_transform(data.values)
    data = pd.DataFrame(scaled_features, columns=data.columns)
    return data

def min_max_scaler(data):
    scaled_features = MinMaxScaler().fit_transform(data.values)
    data = pd.DataFrame(scaled_features, columns=data.columns)
    return data

def robust_scaler(data):
    scaled_features = RobustScaler().fit_transform(data.values)
    data = pd.DataFrame(scaled_features, columns=data.columns)
    return data

def normalizer(data):
    scaled_features = Normalizer().fit_transform(data.values)
    data = pd.DataFrame(scaled_features, columns=data.columns)
    return data
    

def run_pca(data, num_components=2):
    ''' runs the PCA Algorithm'''
    pca = PCA(num_components)
    principal_components = pca.fit_transform(data.values)
    data = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
    print(pca.explained_variance_ratio_)
    return data

def run_clustering(data, iteration_count, n_clusters=3, max_iter=300, tol=0.0001):
    ''' Runs clustering Algorithm'''
    candidate_clusters = []
    errors = []
    for _ in range(iteration_count):
        clusters = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol).fit(data.values)
        error = clusters.inertia_ / len(data) 
        candidate_clusters.append(clusters)
        errors.append(error)
    highest_error = max(errors)
    lowest_error = min(errors)
    print ("Lowest error found:%.2f, Highest error found:%.2f" %(lowest_error, highest_error))
    index_of_lowest_error = errors.index(lowest_error)
    clusters = candidate_clusters[index_of_lowest_error]
    print(clusters.cluster_centers_)
    data['labels'] = clusters.labels_
    return data

def run_gmm(data, n_components=3, covariance_type="diag"):
    gmm = GMM(n_components=3,covariance_type="diag").fit(data)
    labels = gmm.predict(data)
    plt.scatter(data.ix[:, 0], data.ix[:, 1], c=labels, s=40, cmap='viridis');  
    return labels

def variance_threshold_selector(data, threshold=0.5):
     selector = VarianceThreshold(threshold)
     selector.fit(data)
     return data[data.columns[selector.get_support(indices=True)]]



def run_gap_stats(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    Note: The optimal K is the K for which there is higher the gap value 
    """

    gaps = np.zeros((len(range(1, maxClusters)), ))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for (gap_index, k) in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap},
                ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


def main():
    if len(sys.argv) < 2:
        print("Too few arguments")
        sys.exit(1)
    data = read_data_to_dataframe(sys.argv[1])
    import pudb; pudb.set_trace()
    transformed_data = transform_data(data)
    scaled_data = std_scaler(transformed_data)
    clustered_data = run_clustering(scaled_data, 300)
    cluster_labels = clustered_data['labels']
    label_series = pd.Series(cluster_labels)
    print(label_series.value_counts())
    print(run_gap_stats(scaled_data))

if __name__ == "__main__":
    main()
    sys.exit(1)
