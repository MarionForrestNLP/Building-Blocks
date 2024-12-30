# kMeans Clustering Model

# Dependencies

- Imports the `choices` function from the standard `random` module
- Imports the `Callable` class from the standard `typing` module

# Classes

## Distance Functions

This model seeks to assign vectors to a centroid of minimal distance, i.e. min(dist(v, c)). This implementation supports the following distance metrics for calculating the distance between a vector $v$ and a centroid $c$ each with $d$ dimensions.

### Methods

**Euclidean Between** : Calculates the square root of the sum of the squared differences between the corresponding elements of two vectors. Ranges from 0 to infinity. Higher values indicate a greater distance between the vectors.

```math
dist(v, c) = \sqrt{ \sum_{i=1}^d (v_i - c_i)^2 }
```

Performs best with continuous nominal datasets.


    Parameters
    ----------
    aVector : list[int|float]
        The first vector.
    bVector : list[int|float]
        The second vector.

    Returns
    -------
    float
        The Euclidean distance between the two vectors.

    Raises
    ------
    ValueError
        If the vectors do not have the same dimensionality.

**Cosine Between** : Calulates the dot product divided by the product of the magnitudes of two vectors, then subtracts the result from 1. Ranges from 0 to 2. Higher values indicate a greater distance between the vectors.

```math
dist(v, c) = 1 - \frac{ \sum_{i=1}^d v_i c_i }{ \sqrt{\sum_{i=1}^d v_i^2} \sqrt{\sum_{i=1}^d c_i^2} }
```

Performs best when each scalar of a vector represents the presence or occurrence of a unique category, such as a dataset of documents where each word is a category and each scalar indicates the frequency of the word in the document. For example, if 1 indicates the presence of a category and 0 indicates the absence of a category, Cosine distance will perform well on datasets similar to the following:

| id | Category 1 | Category 2 | Category 3 | Category 4 |
|----|------------|------------|------------|------------|
| 1  | 1          | 0          | 0          | 1          |
| 2  | 0          | 1          | 0          | 0          |
| 3  | 0          | 1          | 1          | 0          |
| 4  | 1          | 0          | 0          | 1          |
| 5  | 0          | 1          | 1          | 1          |

or if scalars are dicrete and represent the number of occurrences of a category, Cosine distance will perform well on datasets similar to the following:

| id | Category 1 | Category 2 | Category 3 | Category 4 |
|----|------------|------------|------------|------------|
| 1  | 2          | 0          | 0          | 6          |
| 2  | 0          | 4          | 0          | 0          |
| 3  | 0          | 5          | 1          | 0          |
| 4  | 3          | 0          | 0          | 5          |
| 5  | 0          | 4          | 1          | 2          |


    Parameters
    ----------
    aVector : list[int|float]
        The first vector.
    bVector : list[int|float]
        The second vector.

    Returns
    -------
    float
        The cosine distance between the two vectors.

    Raises
    ------
    ValueError
        If the vectors do not have the same dimensionality.

**Manhattan Between** : Calculates the sum of the absolute differences between the corresponding elements of two vectors. Ranges from 0 to infinity. Higher values indicate a greater distance between the vectors.

```math
dist(v, c) = \sum_{i=1}^d |v_i - c_i|
```

Performs best when each scalar of a vector is either ordinal, 

    Parameters
    ----------
    aVector : list[int|float]
        The first vector.
    bVector : list[int|float]    
        The second vector.

    Returns
    -------
    float
        The Manhattan distance between the two vectors.

    Raises
    ------
    ValueError
        If the vectors do not have the same dimensionality.

**Chebyshev Between** : Calculates the maximum of the absolute differences between the corresponding elements of two vectors. Ranges from 0 to infinity. Higher values indicate a greater distance between the vectors.

```math
dist(v, c) = \max_{i=1}^d |v_i - c_i|
```

Performs best with lower dimensionality. Best used when scalar values are discrete and nominal, and the variance is low.


    Parameters
    ----------
    aVector : list[int|float]
        The first vector.
    bVector : list[int|float]
        The second vector.

    Returns
    -------
    float
        The Chebyshev distance between the two vectors.

    Raises
    ------
    ValueError
        If the vectors do not have the same dimensionality.

## Cluster

Represents a cluster of vectors

### Attributes

- Name ( `str` ) : The name of the cluster.
- Centroid ( `list[ int | float ]` ) : The arithmetic mean vector of the cluster.
- Vectors (` list [ list [ int | float ] ]` ) : The vectors of the cluster.

### Methods

**Recalculate Centroid** : Recalculate the centroid of the cluster according to the centroid strategy. If there are no vectors, return True.

    Parameters
    ----------
    centroidStrategy : str
        The strategy to use for recalculating the centroid.
    distanceFunction : Callable
        The distance function to use for calculating the distance between vectors for median centroids.

    Returns
    -------
    bool
        True if the centroid did not change, False if it did.

## kMeans

### Attributes

**trainingMatrix** ( `list[ list[ int | float ] ]` ): The matrix of vectors to train on.

**kGroups** ( `int` ): The number of clusters to group the data into. Defaults to 3.

**distanceStrategy** ( `str` ): The strategy to use for assigning vectors to clusters. Defaults to "euclidean".
- "euclidean": Use the Euclidean distance metric.
- "cosine": Use the Cosine distance metric.
- "manhattan": Use the Manhattan distance metric. Assigns vectors to the cluster with the most similar distribution of scalars. For example...
- "chebyshev": Use the Chebyshev distance metric.

**centroidStrategy** ( `str` ) : The strategy to use for recalculating the centroid of a cluster. Defaults to "mean".
- "mean": Assign centroids to the average of a given cluster's vectors. Can be affected by outliers. Computes in $O(n)$ time.
- "median": Assign centroids to the median of a given cluster's vectors. More resistant to outliers, but at a cost of increased run time, $O(n^2)$.

**maxEpochs** ( `int` ): The maximum number of epochs to train for. Defaults to 100.

**clusters** ( `list[ Cluster ]` ): The list of clusters. Defaults to an empty list.

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

# Time Complexity

The worst case run time is $O( m(k + kn + kn) )$ for **mean** centroids and $O( m(k + kn + kn^2) )$ for **median** centroids.

- $k$ = number of clusters to group the data into
- $m$ = maximum allowed epochs
- $n$ = number of vectors in the training matrix