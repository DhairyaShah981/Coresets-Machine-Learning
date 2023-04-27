import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
import timeit
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate datasets of different sizes
sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
datasets = {}
for size in sizes:
    X = np.random.rand(size, 10)
    y = np.random.rand(size)
    datasets[size] = (X, y)

# Initialize models
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
kmeans = KMeans(n_clusters=5)

# Measure time taken for each algorithm on each dataset
linear_regression_times = []
logistic_regression_times = []
kmeans_times = []
for size in sizes:
    X, y = datasets[size]
    
    # Time for linear regression
    start_time = timeit.default_timer()
    linear_regression.fit(X, y)
    end_time = timeit.default_timer()
    linear_regression_times.append(end_time - start_time)

    # Time for k-means
    start_time = timeit.default_timer()
    kmeans.fit(X)
    end_time = timeit.default_timer()
    kmeans_times.append(end_time - start_time)


c_datasets = {}
for size in sizes:
    X, y = make_classification(n_samples=size, n_features=10, n_informative=5, n_redundant=0, n_classes=2, random_state=42)
    c_datasets[size] = (X, y)

for size in sizes:
    X, y = c_datasets[size]
    
    # Time for logistic regression
    start_time = timeit.default_timer()
    logistic_regression.fit(X, y)
    end_time = timeit.default_timer()
    logistic_regression_times.append(end_time - start_time)

# Plot results
fig, ax1 = plt.subplots()

ax1.plot(sizes, linear_regression_times, label='Linear Regression')
ax1.plot(sizes, logistic_regression_times, label='Logistic Regression')
ax1.plot(sizes, kmeans_times, label='K-Means')
ax1.set_xlabel('Dataset size')
ax1.set_ylabel('Time taken (seconds)')
ax1.tick_params(axis='y')

fig.legend()
fig.tight_layout()
fig.savefig('training_time.png')
plt.show()