# kMeans Clustering Model

# Classes

## Cluster

Represents a cluster of vectors

### Attributes

- Name ( str ) : The name of the cluster. Used for identification purposes.
- Centroid ( list[ int | float ] ) : The arithmetic mean vector of the cluster.
- Vectors ( list [ list [ int | float ] ] ) : The vectors of the cluster.

### Methods

**Recalculate Centroid** : Recalculate the centroid of the cluster by taking the average of all assigned vectors. If there are no vectors, return the unchanged centroid.

    Returns
    -------
    bool
        True if the centroid did not change, False if it did.

## kMeans

### Attributes

- trainingMatrix ( list[ list[ int | float ] ] ): The matrix of vectors to train on.
- kGroups ( int ): The number of clusters to group the data into.
- strategy ( str ): The strategy to use for assigning vectors to clusters.
- maxEpochs ( int ): The maximum number of epochs to train for.
- clusters ( list[ Cluster ] ): The list of clusters.
- normalized ( bool ): Whether or not the `trainingMatrix` was normalized during the training process.

### Methods

**Train** : Train the model and return the number of epochs trained for.

    Returns
    -------
    int
        The number of epochs trained for.

**Predict** : Predict the cluster a given vector belongs to.

    Parameters
    ----------
    vector : list[ int | float ]
        The vector to predict the cluster for.

    Returns
    -------
    Cluster
        The predicted cluster the vector belongs to.



# Notes

## Time Complexity

Model training takes O(D * N * K * E) time in the worst case.

- E = maximum allowed epochs
- K = number of clusters to group the data into
- N = number of vectors in the training matrix
- D = number of dimensions in the training matrix

## Distance Calculations

This model seeks to assign vectors to a centroid of minimal distance, i.e. min(dist(v, c)). This implementation supports the following distance metrics for calculating the distance between two a vector $v$ and a centroid $c$ each with $d$ dimensions.

### Euclidean Distance (Default)

The Euclidean distance between two vectors is the square root of the sum of the squared differences between the corresponding elements of the vectors.

```math
dist(v, c) = \sqrt{ \sum_{i=1}^d (v_i - c_i)^2 }
```

### Cosine Distance

The cosine distance between two vectors is one minus the cosine of the angle between the vectors. The cosine of the angle between the vectors is the dot product of the vectors divided by the product of the magnitudes of the vectors. Ranges from 0 to 1.

```math
dist(v, c) = 1 - \frac{ \sum_{i=1}^d v_i c_i }{ \sqrt{\sum_{i=1}^d v_i^2} \sqrt{\sum_{i=1}^d c_i^2} }
```

### Manhattan Distance

The Manhattan distance between two vectors is the sum of the absolute differences between the corresponding elements of the vectors.

```math
dist(v, c) = \sum_{i=1}^d |v_i - c_i|
```

### Chebyshev Distance

The Chebyshev distance between two vectors is the maximum of the absolute differences between the corresponding elements of the vectors.

```math
dist(v, c) = \max_{i=1}^d |v_i - c_i|
```