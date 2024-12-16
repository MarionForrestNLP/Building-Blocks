from random import choices
from typing import Callable

class Cluster:
    """
    A class representing a cluster in a clustering algorithm.

    Attributes
    ----------
    name : str
        The name of the cluster.
    centroid : list[int|float]
        The centroid of the cluster.
    vectors : list[list[int|float]]
        A list of vectors assigned to the cluster.

    Methods
    -------
    Recalculate_Centroid() -> bool
        Recalculate the centroid of the cluster.
    """

    def __init__(self, name: str, centroid: list[int|float] = []):
        """
        Initialize a cluster with a name and centroid.
        
        Parameters
        ----------
        name : str
            The name of the cluster.
        centroid : list[int|float], optional
            The centroid of the cluster.
        """
        self.name = name
        self.centroid = centroid
        self.vectors: list[list[int|float]] = []

    def _Average_Vector(self, vectors: list[list[int|float]]) -> list[int|float]:
        """
        Calculate the average vector from a list of vectors.

        Parameters
        ----------
        vectors : list[list[int|float]]
            A list of vectors to average.

        Returns
        -------
        list[int|float]
            The average vector.
        """

        dimensionality: int = len(vectors[0])
        sumVector: list[int|float] = [0] * dimensionality

        for vector in vectors:
            for i in range(dimensionality):
                sumVector[i] += vector[i]

        for i in range(dimensionality):
            sumVector[i] = round((sumVector[i] / len(vectors)), 4)

        return sumVector
    
    def Recalculate_Centroid(self) -> bool:
        # If there are no vectors, return the unchanged centroid
        """
        Recalculate the centroid of the cluster by taking the average of all
        assigned vectors. If there are no vectors, return the unchanged centroid.

        Returns
        -------
        bool
            True if the centroid did not change, False if it did.
        """

        if self.vectors == []: return True

        previousCentroid: list[int|float] = self.centroid
        self.centroid = self._Average_Vector(self.vectors)

        return previousCentroid == self.centroid
    
    def __dict__(self) -> dict:
        return {
            "name": self.name,
            "centroid": self.centroid,
            "vectors": self.vectors
        }
    
    def __str__(self) -> str:
        return f"Cluster: {self.name}, Centroid: {self.centroid}, Vectors: {self.vectors}"
    
class kMeans:
    """
    kMeans model for clustering vectors.

    Attributes
    ----------
    trainingMatrix : list[ list[ int | float ] ]
        The matrix of vectors to train on.
    
    kGroups : int
        The number of clusters to group the data into.
    
    strategy : str
        The strategy to use for assigning vectors to clusters.
    
    maxEpochs : int
        The maximum number of epochs to train for.

    clusters : list[ Cluster ]
        A list of clusters.

    Methods
    -------
    Train()
        Train the model.

    Predict(vector: list[int|float])
        Predict the cluster for a given vector.
    """

    def __init__(self,
            trainingMatrix:list[list[int|float]], 
            kGroups: int = 3,
            strategy: str = "euclidean",
            maxEpochs: int = 100
        ) -> None:
        """
        Initialize a kMeans model with a training matrix and hyperparameters.

        Parameters
        ----------
        trainingMatrix : list[ list[ int | float ] ]
            The matrix of vectors to train on.
        
        kGroups : int, optional
            The number of clusters to group the data into. (Default is 3).
        
        strategy : str, optional
            The strategy to use for assigning vectors to clusters.\n
            "euclidean": Assign vectors to the cluster with the lowest *Euclidean* distance.\n
            "cosine": Assign vectors to the cluster with the highest *Cosine* similarity.\n
            "manhattan": Assign vectors to the cluster with the lowest *Manhattan* distance.\n
            "chebyshev": Assign vectors to the cluster with the lowest *Chebyshev* distance.\n
            (Default is "euclidean").
        
        maxEpochs : int, optional
            The maximum number of epochs to train for. (Default is 100).
        """

        self.trainingMatrix: list[list[int|float]] = trainingMatrix
        self.kGroups: int = kGroups
        self.maxEpochs: int = maxEpochs
        self.strategy = strategy

        self.clusters: list[Cluster] = []
    
    def _Initialize_Clusters(self) -> list[Cluster]:
        """
        Initialize clusters with random vectors from the training matrix.

        Returns
        -------
        list[Cluster]
            A list of initialized clusters.
        """
        clusters: list[Cluster] = []
        randomVectors: list[list[int|float]] = choices(self.trainingMatrix, k=self.kGroups)

        for i in range(self.kGroups):
            clusters.append(
                Cluster(
                    name=f"cluster_{i}",
                    centroid=randomVectors[i]
                )
            )

        return clusters
    
    def _Validate_Dimensionality(self, aVector: list[int|float], bVector: list[int|float]) -> None:
        """
        Validates that two vectors have the same dimensionality.

        Parameters
        ----------
        aVector : list[ int | float ]
            The first vector.
        bVector : list[ int | float ]
            The second vector.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """

        if len(aVector) != len(bVector):
            raise ValueError(f"Vectors must have the same dimensionality: {len(aVector)} != {len(bVector)}")

    def _Cosine_Similarity(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculates the cosine similarity between two vectors and returns the "*cosine distance*".

        Parameters
        ----------
        aVector : list[ int | float ]
            The first vector.
        bVector : list[ int | float ]
            The second vector.

        Returns
        -------
        float
            The Cosine Similarity of the two vectors. Ranges from 0 to 1.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """
        
        self._Validate_Dimensionality(aVector, bVector)

        dotProduct: float = sum(a * b for a, b in zip(aVector, bVector))
        magnitudeA: float = sum(a * a for a in aVector) ** 0.5
        magnitudeB: float = sum(b * b for b in bVector) ** 0.5

        return 1 - (dotProduct / (magnitudeA * magnitudeB))
    
    def _Euclidean_Between(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculates the Euclidean distance between two vectors.

        Parameters
        ----------
        aVector : list[ int | float ]
            The first vector.
        bVector : list[ int | float ]
            The second vector.

        Returns
        -------
        float
            The Euclidean distance between the two vectors.

        Raises
        ------
        ValueError
            If the vectors do not have the same dimensionality.
        """
        
        self._Validate_Dimensionality(aVector, bVector)

        distance: float = 0.0
        preRootSum: float = 0.0

        # Sum of squared differences
        for i in range(len(aVector)):
            preRootSum += (aVector[i] - bVector[i])**2

        # Square root of the sum of squared differences
        distance = round((preRootSum**0.5), 4)

        return distance
    
    def _Manhattan_Between(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculate the Manhattan distance between two vectors.

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
        """

        self._Validate_Dimensionality(aVector, bVector)

        distance: float = 0.0

        for i in range(len(aVector)):
            distance += abs(aVector[i] - bVector[i])

        return distance
    
    def _Chebyshev_Between(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        """
        Calculate the Chebyshev distance between two vectors.

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
        """

        self._Validate_Dimensionality(aVector, bVector)

        distance: float = 0.0

        for i in range(len(aVector)):
            distance = max(distance, abs(aVector[i] - bVector[i]))

        return distance
    
    def _Strategy_Function(self) -> Callable:
        """
        Returns a tuple with the function used for calculating the **fit** of a centroid
        and the optimization function to use to find the best fitting centroid.

        Returns
        -------
        tuple[ callable ]
            A tuple with the fit function (0) and the optimization function (1).
        """

        if self.strategy == "euclidean":
            return self._Euclidean_Between
        elif self.strategy == "cosine":
            return self._Cosine_Similarity
        elif self.strategy == "manhattan":
            return self._Manhattan_Between
        elif self.strategy == "chebyshev":
            return self._Chebyshev_Between
        
        else:
            return self._Euclidean_Between

    def Train(self) -> int:
        """
        Trains the kMeans model.

        Returns
        -------
        int
            The number of epochs it took to converge.
        """

        # Initialize clusters
        self.clusters = self._Initialize_Clusters()

        # Initialize variables
        epoch: int = 0
        isConverged: bool = False
        stratFunc = self._Strategy_Function()

        while not isConverged and epoch < self.maxEpochs:
            # Clear the clusters
            for cluster in self.clusters: cluster.vectors = []

            # Assign vectors to clusters
            for vector in self.trainingMatrix:
                # Calculate centroid separations
                centroidSeparations: dict[int, float] = {
                    centroid: stratFunc(vector, cluster.centroid) for centroid, cluster in enumerate(self.clusters)
                }

                # Get the closest centroid
                closestCentroid: int = min(centroidSeparations, key=centroidSeparations.get)

                # Add the vector to the closest centroid
                self.clusters[closestCentroid].vectors.append(vector)

            # Calculate new centroids and check for convergence
            convergenceArray: list[bool] = [
                cluster.Recalculate_Centroid() for cluster in self.clusters
            ]
            isConverged = all(convergenceArray)

            # Increment epoch
            epoch += 1

        return epoch
    
    def Predict(self, vector: list[int|float]) -> Cluster:
        """
        Predicts the cluster to which a given vector belongs.

        Parameters
        ----------
        vector : list[ int | float ]
            The vector to predict the cluster for.

        Returns
        -------
        Cluster
            The predicted cluster the vector belongs to.
        """

        stratFunc = self._Strategy_Function()
        centroidSeparations: dict[int, float] = {
            centroid: stratFunc(vector, cluster.centroid) for centroid, cluster in enumerate(self.clusters)
        }
        clusterIndex: int = min(centroidSeparations, key=centroidSeparations.get)
        return self.clusters[clusterIndex]

