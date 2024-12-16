# K Means Classifier

Classes
- [Cluster](#cluster)
- [KMeans](#k-means)

Notes
- [Time Complexity](#time-complexity)
- [Classification Threshold](#classification-threshold)

# Classes

## Cluster
The cluster class represents a classified group of vectors.

### Attributes

- Name ( str ) : The name of the cluster. Used for identification purposes.
- Centroid ( list[ int | float ] ) : The arithmetic mean vector of the cluster.
- Vectors ( list [ list [ int | float ] ] ) : The vectors of the cluster

### Methods

#### Recalculate Centroid

Recalculates the centroid of the cluster by taking the average of all assigned vectors. If there are no vectors, returns the unchanged centroid.

Returns
- list[int|float]: The new (or unchanged) centroid of the cluster.

#### To Dictionary

Converts the cluster object to a dictionary for easyier implementation into your own code.

Returns
- dict: A dictionary with keys "name", "centroid", and "vectors".

## K Means
The K Means class represents a kMeans clustering model. The model is train at initialization, a call to a "train" method is not required.

### Attributes
- trainingMatrix ( list [ list [ int | float ] ] ): The matrix of - vectors to train on.
- kGroups ( int ): The number of clusters to group the data into.
- threshold ( float ): The threshold to stop convergence.
- maxEpochs ( int ): The maximum number of epochs to train for.
- clusters ( list [ Cluster ] ): The list of clusters.
- unassignedVectors ( list [ list [ int | float ] ] ): The list of - vectors that were not assigned to a cluster.
- epoch ( int ): The number of epochs trained for.

### Methods

**Fine Tine** : Retrain the model with new hyperparameters.

Parameters
- kGroups ( int ): The new number of clusters to group the data into.
- threshold ( float ): The new threshold to stop convergence.
- maxEpochs ( int ): The new maximum number of epochs to train for.

Returns
- tuple: A tuple of the new clusters, the new unassigned vectors, and the new number of epochs trained for.

**Predict** : Predict the cluster a given vector belongs to. Returns None if no cluster is found within the model's classification threshold. The model can also be retrained using the given vector.

Parameters
- vector ( list [ int | float ] ): The vector to classify.
- retrain ( bool ): Whether to retrain the model using the new vector. Defaults to False.

Returns
- Cluster | None: The predicted cluster or None if no cluster was found.

# Notes

## Time Complexity

Model training takes O((n * d) + (n * k * e)) time in the worst case.

- n = number of vectors in the training matrix
- d = number of dimensions in the matrix
- k = number of clusters being identified
- e = maximum allowed epochs

## Classification Threshold

Whether a vector is grouped into a given cluster is determined by the following principle:

```
m = a matrix of vectors
v = a given vector in m
s = a given cluster
c = the centroid of s

v is in s if:
    dist(v, c) < max_dist(m) * T
```

**dist(v, c)** : The euclidean distance between vector v and centroid c.

```
v = a vector with i dimensions
c = a centroid with i dimensions

distance(v, c) = sqrt( (v[1] - c[1])^2 + (v[2] - c[2])^2 + ... + (v[i] - c[i])^2 )
```

**max_dist(m)** : The maximum possible distance between any two vectors in matrix m. This is calculated using a formula jointedly discovered by RJ Clines and Marion Forrest, two undergraduate students at the University of North Florida.
```
m = a matrix of vectors
l = largest scalar in m
s = smallest scalar in m
d = the number of dimensions in m

max_distance(m) = (l - s) * sqrt( d )
```

**T** : The value of the `threshold` parameter of the kMeans class. Ranges from 0 to 1. Defaults to 0.1

*Example*
```
m = [
    [1, 1, 3],
    [4, -5, 2],
    [7, -8, 0]
]
v = [4, -5, 2]
s = Cluster{}
c = [4, 8, 9]

v is in s if:
    dist(v, c) < max_dist(m) * T
    \/
    14.7648 < 25.9807 * 0.1
    \/
    14.7648 < 2.5980
    \/
    False, v is not in s
```