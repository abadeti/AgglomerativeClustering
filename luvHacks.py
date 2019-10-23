import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import matplotlib
    
def preprocessing(file_before_preprocessing):
    data = pd.read_csv(file_before_preprocessing)
    panda_file = pd.DataFrame(data)
    og_panda_file = panda_file.iloc[:,1:]
    panda_file = panda_file.iloc[:,2:]
    for col in panda_file:
        panda_file[col] = panda_file[col].str[0]
    panda_file.to_csv('luvHacksResults.csv')
    return([panda_file, og_panda_file])
    
def agglomerative_clustering(out_path, pd_data, og_file, number_of_clusters):
    agglomerative_clustering = AgglomerativeClustering(n_clusters=number_of_clusters, linkage="complete")
    agglomerative_clustering.fit(pd_data)

    # form clusters
    clusters = {}
    for i in range(0, number_of_clusters):
        if i not in clusters:
            clusters[i] = []
        for j in range(0, len(agglomerative_clustering.labels_)):  # a label for each repo.
            if i == agglomerative_clustering.labels_[j]:  # if repo label is equal to Cluster number
                clusters[i].append(og_file["What's Your Name?"][j])  # add repo to cluster i's list.

    clusters_output = pd.DataFrame.from_dict(clusters, orient='index')
    clusters_output.to_csv(out_path)
        
def dend(panda_file):
    df = panda_file[1].set_index("What's Your Name?")
    del df.index.name
    Z = hierarchy.linkage(df, 'complete')
    hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)
    matplotlib.pyplot.gcf()
    matplotlib.pyplot.savefig("dendrogram.png", dpi=320, bbox_inches='tight')


if __name__ == '__main__':
    panda = preprocessing("LuvHacks.csv")
    agglomerative_clustering("luvHacksClusters.csv", panda[0], panda[1], 5)
    dend(panda)