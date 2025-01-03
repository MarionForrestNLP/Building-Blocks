# K-Clusters Model

[Dependencies](#dependencies)

[Classes](#classes)
- [Cluster](#cluster)
- [K-Clusters](#k-clusters)

[Distance Metrics](#distance-metrics)
- [Euclidean Distance](#euclidean-distance)
- [Cosine Distance](#cosine-distance)
- [Manhattan Distance](#manhattan-distance)

[Time Complexity](#time-complexity)

# Dependencies

- Imports the `choices` function from the standard `random` module
- Imports the `Callable` class from the standard `typing` module

# Classes

## Cluster

Represents a cluster of vectors

### Attributes

- **Name** ( `str` ) : The name of the cluster.

- **Centroid** ( `list[ int | float ]` ) : A vector that represents the central point of the cluster.

- **Vectors** (` list [ list [ int | float ] ]` ) : The vectors of the cluster.

- **Silhouette Score** ( `float` ) : The average silhouette score of each vector in the cluster.

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

## K-Clusters

K-Clusters model for clustering vectors with either mean or median centroids.

### Attributes

- **Training Matrix** ( `list[ list[ int | float ] ]` ): The matrix of vectors to train on.

- **K Groups** ( `int` ): The number of clusters to group the data into. Defaults to 3.

- **Distance Strategy** ( `str` ): The strategy to use for assigning vectors to clusters. Defaults to "euclidean".
    - "euclidean": Use the [Euclidean](#euclidean-distance) distance metric.
    - "cosine": Use the [Cosine](#cosine-distance) distance metric.
    - "manhattan": Use the [Manhattan](#manhattan-distance) distance metric.

- **Centroid Strategy** ( `str` ) : The strategy to use for recalculating the centroid of a cluster. Defaults to "mean".
    - "mean": Assign centroids to the average of a given cluster's vectors. Computes in $O(n)$ time.
    - "median": Assign centroids to the median of a given cluster's vectors. Performs better with non-euclidean distance metrics. More resistant to outliers, but at a cost of increased run time, $O(n^2)$.

- **Max Epochs** ( `int` ): The maximum number of epochs to train for. Defaults to 100.

- **Clusters** ( `list[ Cluster ]` ): The list of clusters. Defaults to an empty list.

- **Silhouette Score** ( `float` ): The average silhouette score of each cluster in the model.

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

# Distance Metrics

This model seeks to assign a given vector $v$ to a centroid $c$ of minimal distance, i.e. min(dist($v$, $c$)), within $d$ dimensions. This implementation supports the following distance metrics.

## Euclidean Distance

Calculates the square root of the sum of the squared differences between the corresponding elements of two vectors. Ranges from 0 to infinity.

```math
dist(v, c) = \sqrt{ \sum_{i=1}^d (v_i - c_i)^2 }
```

Performs best when each scalar of a vector is either a discrete or continuous numeric value.

*Example Dataset:*
| id | price | # bed rooms | # bathrooms | # stories |
|----|-------|-------------|-------------|-----------|
| 1  | 280,000   | 3       | 2           | 1         |
| 2  | 550,000   | 3       | 3.5         | 3         |
| 3  | 1,300,000 | 4       | 4.5         | 2         |

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

## Cosine Distance

Calulated by dividing the dot product by the product of the magnitudes of two vectors, then subtracts the result from 1. Ranges from 0 to 2.

```math
dist(v, c) = 1 - \frac{ \sum_{i=1}^d v_i c_i }{ \sqrt{\sum_{i=1}^d v_i^2} \sqrt{\sum_{i=1}^d c_i^2} }
```

Performs best when each scalar of a vector represents the presence or occurrence of a unique category, such as a dataset of documents where each word is a unique category and each scalar indicates the frequency of the word in the document. For example, if 1 indicates the presence of a category and 0 indicates the absence of a category, Cosine distance will perform well on datasets similar to the following:

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

## Manhattan Distance

Calculates the sum of the absolute differences between the corresponding elements of two vectors. Ranges from 0 to infinity.

```math
dist(v, c) = \sum_{i=1}^d |v_i - c_i|
```

Performs best when each scalar of a vector is ordinal. When in the distinction between one and three is not an increase of two, but rather the implication that of a new sub-category. For example, a vector may contain 1, 2, and 3, as numerical standins for the distinct sub-categories of aerospace, maritime, and automotive in a dataset of different manufactured vehicles.

This distance metric is also useful when scalar values represent the grouping of contiunuous values. Such a range of salaries may be mapped as such: 0-50000 => 1, 50001-100000 => 2, 100001-150000 => 3, 150001-200000 => 4.

*Example Dataset:*
| id | Industry | Weight Class |
|----|---|---|
| 1  | 1 | 1 |
| 2  | 3 | 2 |
| 3  | 2 | 4 |
| 4  | 1 | 1 |
| 5  | 3 | 4 |
| 6  | 2 | 3 |
| 7  | 1 | 1 |

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

# Time Complexity

The worst case run time is $O( m(k + kn + kn) )$ for **mean** centroids and $O( m(k + kn + kn^2) )$ for **median** centroids.

- $k$ = number of clusters to group the data into
- $m$ = maximum allowed epochs
- $n$ = number of vectors in the training matrix