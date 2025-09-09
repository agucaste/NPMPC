# Test using Meta's FAISS library "https://github.com/facebookresearch/faiss"

import numpy as np
import faiss as fa
from sklearn.neighbors import KDTree
import time
# Parameters
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # number of queries

def test_faiss(seed: int):
    np.random.seed(seed)             # make reproducible

    # Build data and queries
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    start_time = time.time()
    index = fa.IndexFlatL2(d)   # build the index
    index.add(xb)               # add vectors to the index
    creation_time = time.time() - start_time

    k = 3                       # we want to see 3 nearest neighbors

    start_time = time.time()
    D, I = index.search(xq, k)  # actual search
    search_time = time.time() - start_time

    # print(f"FAISS - Index creation time: {creation_time:.4f} seconds")
    # print(f"FAISS - Search time: {search_time:.4f} seconds")
    # print(f"Neighbors of the first five queries:")
    # print(I[:5])                   # neighbors of the 5 first queries
    # print(f"Distances to the first five queries")
    # print(D[:5])                   # distances of the 5 first queries
    return creation_time, search_time

def test_kdtree(seed: int):
    np.random.seed(seed)             # make reproducible

    # Build data and queries
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    start_time = time.time()
    tree = KDTree(xb)  # build the KDTree
    creation_time = time.time() - start_time

    k = 3              # we want to see 3 nearest neighbors

    start_time = time.time()
    dist, ind = tree.query(xq, k)  # actual search
    search_time = time.time() - start_time

    # print(f"KDTree - Tree creation time: {creation_time:.4f} seconds")
    # print(f"KDTree - Search time: {search_time:.4f} seconds")
    return creation_time, search_time
    # print(f"Neighbors of the first five queries:")
    # print(ind[:5])                # neighbors of the 5 first queries
    # print(f"Distances to the first five queries")
    # print(dist[:5])               # distances of the 5 first queries


# Run both tests
N = 100
import matplotlib.pyplot as plt

fa_times_creation, fa_times_query = zip(*[test_faiss(s*12) for s in range(N)])
kdtree_times_creation, kdtree_times_query = zip(*[test_kdtree(s*12) for s in range(N)])

print(f"\nFAISS\n - Creation: {np.mean(fa_times_creation):.4f} ± {np.std(fa_times_creation):.4f} seconds")
print(f" - Query: {np.mean(fa_times_query):.4f} ± {np.std(fa_times_query):.4f} seconds")
print(f"\nKDTree\n - Creation: {np.mean(kdtree_times_creation):.4f} ± {np.std(kdtree_times_creation):.4f} seconds")
print(f" - Query: {np.mean(kdtree_times_query):.4f} ± {np.std(kdtree_times_query):.4f} seconds")


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
bins = int(np.sqrt(N))

# FAISS creation times
axes[0, 0].hist(fa_times_creation, bins=bins, color='C0', alpha=0.7)
axes[0, 0].set_title("FAISS - Creation Times")
axes[0, 0].set_ylabel("Count")

# FAISS query times
axes[0, 1].hist(fa_times_query, bins=bins, color='C0', alpha=0.7)
axes[0, 1].set_title("FAISS - Query Times")

# KDTree creation times
axes[1, 0].hist(kdtree_times_creation, bins=bins, color='C1', alpha=0.7)
axes[1, 0].set_title("KDTree - Creation Times")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_xlabel("Time (seconds)")

# KDTree query times
axes[1, 1].hist(kdtree_times_query, bins=bins, color='C1', alpha=0.7)
axes[1, 1].set_title("KDTree - Query Times")
axes[1, 1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig('faiss-vs-kdtree.pdf')
plt.savefig('faiss-vs-kdtree.png', dpi=300)
plt.show()

