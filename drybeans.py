import pandas as pd
import numpy as np
import zipfile
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from mnist import MNIST
from keras.datasets import mnist
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
import timeit

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images / 255.0
# Reshape test images into vectors
X = pd.DataFrame(test_images.reshape(test_images.shape[0], -1))
true_labels = test_labels

n_samples = X.shape[0]
n_clusters = 10

# # Open the zip file and read the excel file
# with zipfile.ZipFile('DryBeanDataset.zip', 'r') as zip_ref:
#     with zip_ref.open('DryBeanDataset/Dry_Bean_Dataset.xlsx') as file:
#         # Load the data into a pandas dataframe
#         df = pd.read_excel(file)

# # # Split the features and labels
# X = df.iloc[:, :-1]
# true_labels = df.iloc[:, -1]

# # Set the number of clusters
# n_samples = X.shape[0]
# n_clusters = 7

# Fit KMeans on the full dataset
start = timeit.default_timer()
kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 10, random_state = 42).fit(X)
stop = timeit.default_timer()

full_kmeans_labels = kmeans.labels_
full_kmeans_score = silhouette_score(X, full_kmeans_labels, random_state=42)
full_kmeans_inertia = kmeans.inertia_
full_kmeans_rand = rand_score(true_labels, full_kmeans_labels)
full_kmeans_time = stop - start
print("PART 0 DONE")

# # Define the fractional size of the coresets
m_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

def run_uniform_coreset(m): 
    # Generate the uniform coresets
    idxs = np.random.choice(X.index, int(m*n_samples), replace=False)
    coreset = X.loc[idxs]

    # Fit KMeans on the uniform coresets and calculate the scores
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 10, random_state = 42).fit(coreset)
    stop = timeit.default_timer()

    kmeans_labels = kmeans.labels_
    labels = kmeans.predict(X)
    uniform_scores = silhouette_score(X, labels, random_state=42)
    uniform_inertia = kmeans.inertia_
    uniform_rand = rand_score(true_labels, labels)

    return uniform_scores, uniform_inertia, uniform_rand, stop - start

def run_importance_coreset(m):
    # Generate the importance sampling coresets
    weights = np.random.rand(X.shape[0])
    idxs = np.random.choice(X.index, int(m*n_samples), p=weights/np.sum(weights), replace=False)
    coreset = X.loc[idxs]

    # Fit KMeans on the importance sampling coresets and calculate the scores
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 10, random_state = 42).fit(coreset)
    stop = timeit.default_timer()

    kmeans_labels = kmeans.labels_
    labels = kmeans.predict(X)
    importance_scores = silhouette_score(X, labels, random_state=42)
    importance_inertia = kmeans.inertia_
    importance_rand = rand_score(true_labels, labels)

    return importance_scores, importance_inertia, importance_rand, stop - start

def k_centers():
    centers = []
    centers.append(X.iloc[np.random.choice(len(X))]) # First center
    for k in range(1, n_clusters):
        distances = np.zeros(len(X))
        for i in range(len(X)): # Datapoints/Images
            dist = np.inf
            for c in centers:
                # Find the distance of X[i] from each centers we have so far and store the min
                cur_dist = np.linalg.norm(X.iloc[i] - c)
                if(cur_dist < dist):
                    dist = cur_dist
            distances[i] = dist
        
        max_dist = distances[0]
        point = 0
        for j in range(1, len(distances)):
            if(distances[j] > max_dist):
                max_dist = distances[j]
                point = j
                
        centers.append(X.iloc[point])
    return centers

def init():
    centers = []
    centers.append(X.iloc[np.random.choice(len(X))]) # First center

    for k in range(1, n_clusters):
        distances = np.zeros(len(X))
        for i in range(len(X)): # Datapoints/Images
            dist = np.inf
            for c in centers:
                # Find the min distance of X[i] from each centers we have so far and store the min
                cur_dist = np.linalg.norm(X.iloc[i] - c)
                if(cur_dist < dist):
                    dist = cur_dist
            distances[i] = dist

        prob = (distances ** 2) / np.sum(distances ** 2)
        centers.append(X.iloc[np.random.choice(len(X), p = prob)])
    return centers

def find_dist(centers):
    distances = np.zeros(len(X))
    for i in range(len(X)): # Datapoints/Images
        dist = np.inf
        for c in centers:
            cur_dist = np.linalg.norm(X.iloc[i] - c)
            if(cur_dist < dist):
                dist = cur_dist
        distances[i] = dist
    return distances

def find_q():
    init_centers = k_centers()
    distances = find_dist(init_centers) ** 2
    denom = np.sum(distances)
    q = np.zeros(len(X))
    for i in range(len(X)):
        q[i] = distances[i] / denom
    return q

def find_q2():
    init_centers = init()
    distances = find_dist(init_centers) ** 2
    denom = np.sum(distances)
    q = np.zeros(len(X))
    for i in range(len(X)):
        q[i] = distances[i] / denom
    return q

def find_prob(w):
    weighted_data = X * w[:, np.newaxis]
    data_mean = np.sum(weighted_data, axis=0) / np.sum(w)
    prob = np.sum((weighted_data - data_mean[np.newaxis, :]) ** 2, axis=1) * w
    if np.sum(prob) > 0:
        prob = prob / np.sum(prob) * 0.5 + 0.5 / np.sum(w)
    else:
        prob = np.ones(len(X)) / np.sum(w)
    # normalize in order to avoid numerical errors
    prob /= np.sum(prob)

def run_og_importance_coreset(m, prob):
    # Generate the og importance sampling coresets
    idxs = np.random.choice(X.index, int(m*n_samples), p=prob, replace=False)
    coreset = X.loc[idxs]

    # Fit KMeans on the og importance sampling coresets and calculate the scores
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 10, random_state = 42).fit(coreset)
    stop = timeit.default_timer()

    kmeans_labels = kmeans.labels_
    labels = kmeans.predict(X)
    og_scores = silhouette_score(X, labels, random_state=42)
    og_inertia = kmeans.inertia_
    og_rand = rand_score(true_labels, labels)

    return og_scores, og_inertia, og_rand, stop - start

def run_q_importance_coreset(m, prob):
    # Generate the og importance sampling coresets
    idxs = np.random.choice(X.index, int(m*n_samples), p=prob, replace=False)
    coreset = X.loc[idxs]

    # Fit KMeans on the og importance sampling coresets and calculate the scores
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', n_init = 10, random_state = 42).fit(coreset)
    stop = timeit.default_timer()

    kmeans_labels = kmeans.labels_
    labels = kmeans.predict(X)
    qimp_scores = silhouette_score(X, labels, random_state=42)
    qimp_inertia = kmeans.inertia_
    qimp_rand = rand_score(true_labels, labels)

    return qimp_scores, qimp_inertia, qimp_rand, stop - start


w = find_q()
w2 = find_q2()
print("PART 1 DONE")
prob = find_prob(w)
prob2 = find_prob(w2)
print("PART 2 DONE")

uniform_scores = []
uniform_inertia = []
uniform_rand = []
uniform_time = []
importance_scores = []
importance_inertia = []
importance_rand = []
importance_time = []
og_scores = []
og_inertia = []
og_rand = []
og_time = []
means_scores = []
means_inertia = []
means_rand = []
means_time = []
for m in tqdm(m_list):
    u_scores, u_inertia, u_rand, u_time = run_uniform_coreset(m)
    i_scores, i_inertia, i_rand, i_time = run_importance_coreset(m)
    o_scores, o_inertia, o_rand, o_time = run_og_importance_coreset(m, prob)
    m_scores, m_inertia, m_rand, m_time = run_q_importance_coreset(m, prob2)
    uniform_scores.append(u_scores)
    uniform_inertia.append(u_inertia)
    uniform_rand.append(u_rand)
    uniform_time.append(u_time)
    importance_scores.append(i_scores)
    importance_inertia.append(i_inertia)
    importance_rand.append(i_rand)
    importance_time.append(i_time)
    og_scores.append(o_scores)
    og_inertia.append(o_inertia)
    og_rand.append(o_rand)
    og_time.append(o_time)
    means_scores.append(m_scores)
    means_inertia.append(m_inertia)
    means_rand.append(m_rand)
    means_time.append(m_time)

print("PART 3 DONE")

def plot():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 5))
    ax1.plot(m_list, uniform_scores, label='Uniform Coreset')
    ax1.plot(m_list, importance_scores, label='Random Importance Sampling Coreset')
    ax1.plot(m_list, og_scores, label='OG Importance Sampling Coreset')
    ax1.plot(m_list, means_scores, label='Importance Sampling Coreset')
    ax1.plot(m_list, [full_kmeans_score]*len(m_list), label='KMeans')
    ax1.set_xlabel('Coreset Size')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Coreset Scores')
    ax1.legend()

    ax2.plot(m_list, uniform_inertia, label='Uniform Coreset')
    ax2.plot(m_list, importance_inertia, label='Random Importance Sampling Coreset')
    ax2.plot(m_list, og_inertia, label='OG Importance Sampling Coreset')
    ax2.plot(m_list, means_inertia, label='Importance Sampling Coreset')
    ax2.plot(m_list, [full_kmeans_inertia]*len(m_list), label='KMeans')
    ax2.set_xlabel('Coreset Size')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Coreset Inertia')
    ax2.legend()

    ax3.plot(m_list, uniform_rand, label='Uniform Coreset')
    ax3.plot(m_list, importance_rand, label='Random Importance Sampling Coreset')
    ax3.plot(m_list, og_rand, label='OG Importance Sampling Coreset')
    ax3.plot(m_list, means_rand, label='Importance Sampling Coreset')
    ax3.plot(m_list, [full_kmeans_rand]*len(m_list), label='KMeans')
    ax3.set_xlabel('Coreset Size')
    ax3.set_ylabel('Rand Score')
    ax3.set_title('Coreset Rand Score')
    ax3.legend()

    ax4.plot(m_list, uniform_time, label='Uniform Coreset')
    ax4.plot(m_list, importance_time, label='Random Importance Sampling Coreset')
    ax4.plot(m_list, og_time, label='OG Importance Sampling Coreset')
    ax4.plot(m_list, means_time, label='Importance Sampling Coreset')
    ax4.plot(m_list, [full_kmeans_time]*len(m_list), label='KMeans')
    ax4.set_xlabel('Coreset Size')
    ax4.set_ylabel('Training Time')
    ax4.set_title('Coreset Training Time')
    ax4.legend()

    plt.savefig('performance_beans.png')
    plt.show()

plot()
print("DONE")

# Print the results
# print('KMeans score:', full_kmeans_score)
# print('Median Uniform coreset scores:', uniform_scores)
# print('Median Importance coreset scores:', importance_scores)
# print('Median OG Importance coreset scores:', og_scores)

# print('KMeans inertia:', full_kmeans_inertia)
# print('Median Uniform coreset inertia:', uniform_inertia)
# print('Median Importance coreset inertia:', importance_inertia)
# print('Median OG Importance coreset inertia:', og_inertia)
