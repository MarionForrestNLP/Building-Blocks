from random import randint, choice

class Cluster:
    def __init__(
        self, 
        name: str, 
        centroid: list[int|float] = []
    ):
        """Initialize a cluster with its name, centroid, and vectors."""

        self.name = name
        self.centroid = centroid
        self.vectors: list[list[int|float]] = []

    def Recalculate_Centroid(self) -> list[int|float]:
        """
        Recalculates the centroid of the cluster by taking the average of all
        assigned vectors. If there are no vectors, returns the unchanged centroid.

        Returns:
            list[int|float]: The new centroid of the cluster.
        """
        
        # If there are no vectors, return the unchanged centroid
        if self.vectors == []:
            return self.centroid

        vectorCount: int = len(self.vectors)
        dimensionCount: int = len(self.vectors[0])

        # Sum of vectors
        sumVector: list[int|float] = [0] * dimensionCount
        for vector in self.vectors:
            for i in range(dimensionCount):
                sumVector[i] += vector[i]

        # Average of vectors
        for i in range(dimensionCount):
            sumVector[i] = round((sumVector[i] / vectorCount), 4)

        # Reassign the centroid
        self.centroid = sumVector
        return sumVector
        
    def To_Dict(self) -> dict[str, str | list[int|float] | list[list[int|float]]]:
        """
        Convert the cluster object to a dictionary format.

        Returns:
            dict: A dictionary with keys "name", "centroid", and "vectors".
        """

        return {
            "name": self.name,
            "centroid": self.centroid,
            "vectors": self.vectors
        }    
class K_Means:
    def __init__(self, trainingMatrix:list[list[int|float]], kGroups: int = 3, threshold: float = 0.1, maxEpochs: int = 100):
        """
        Initialize a K-Means model with a training matrix and hyperparameters.

        Parameters:
            trainingMatrix (list[list[int|float]]): The matrix of vectors to train on.
            kGroups (int): The number of clusters to group the data into. Defaults to 3.
            threshold (float): The threshold to stop convergence. Defaults to 0.1.
            maxEpochs (int): The maximum number of epochs to train for. Defaults to 100.

        Attributes:
            trainingMatrix (list[list[int|float]]): The matrix of vectors to train on.
            kGroups (int): The number of clusters to group the data into.
            threshold (float): The threshold to stop convergence.
            maxEpochs (int): The maximum number of epochs to train for.
            clusters (list[Cluster]): The list of clusters.
            unassignedVectors (list[list[int|float]]): The list of vectors that were not assigned to a cluster.
            epoch (int): The number of epochs trained for.
        """
        
        self.trainingMatrix: list[list[int|float]] = trainingMatrix
        self.kGroups: int = kGroups
        self.threshold: float = threshold
        self.maxEpochs: int = maxEpochs
        self.clusters: list[Cluster] = []
        self.unassignedVectors: list[list[int|float]] = []
        self.epoch: int = 0

        # Clamp threshold
        if self.threshold < 0:
            self.threshold = 0
        if self.threshold > 1:
            self.threshold = 1
        
        # Train the model
        self.clusters, self.unassignedVectors, self.epoch = self._Train(
            matrix=self.trainingMatrix,
            k=self.kGroups,
            threshold=self.threshold,
            maxEpochs=self.maxEpochs
        )

    def Fine_Tune(self, kGroups:int, threshold:float, maxEpochs:int) -> tuple[list[Cluster], list[list[int|float]], int]:
        """
        Fine tune the model by retraining with new hyperparameters.

        Parameters:
            kGroups (int): The new number of clusters to group the data into.
            threshold (float): The new threshold to stop convergence.
            maxEpochs (int): The new maximum number of epochs to train for.

        Returns:
            tuple: A tuple of the new clusters, the new unassigned vectors, and the new number of epochs trained for.
        """

        self.kGroups = kGroups
        self.threshold = threshold
        self.maxEpochs = maxEpochs

        self.clusters, self.unassignedVectors, self.epoch = self._Train(
            matrix=self.trainingMatrix,
            k=self.kGroups,
            threshold=self.threshold,
            maxEpochs=self.maxEpochs
        )
    
    def Predict(self, vector: list[int|float], retrain: bool = False) ->tuple[list[int|float], Cluster | None]:
        """
        Predict the cluster of a given vector.

        Parameters:
            vector (list[int|float]): The vector to classify.
            retrain (bool): Whether to retrain the model using the new vector. Defaults to False.

        Returns:
            tuple[list[int|float], Cluster | None]: A tuple of the vector and the predicted cluster or None if no cluster was found.
        """
        if retrain:
            temp: list[list[int|float]] = self.trainingMatrix
            temp.append(vector)
            self.trainingMatrix = temp

            self.clusters, self.unassignedVectors, self.epoch = self._Train(
                matrix=self.trainingMatrix,
                k=self.kGroups,
                threshold=self.threshold,
                maxEpochs=self.maxEpochs
            )

        closestDistance: float = Calc_Max_Possible_Distance(self.trainingMatrix) * self.threshold
        closestCluster: Cluster = None

        for clusterObject in self.clusters:
            currentDistance: float = Calc_Distance(aVector=vector, bVector=clusterObject.centroid)

            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestCluster = clusterObject

        if closestCluster == None:
            return (vector, None)
        else:
            return (vector, closestCluster)
        
    def To_Dict(self) -> dict:
        """
        Convert the model to a dictionary.

        Returns:
            dict: A dictionary representation of the model.
        """
        return {
            "clusters": [cluster.To_Dict() for cluster in self.clusters],
            "unassignedVectors": self.unassignedVectors,
            "epoch": self.epoch
        }

    def _Train(self, matrix:list[list[int|float]], k:int, threshold:float, maxEpochs:int) -> tuple[list[Cluster], list[list[int|float]], int]:
        """
        Train the model using a given matrix, number of clusters, threshold, and maximum epochs.

        Parameters:
            matrix (list[list[int|float]]): The matrix of vectors to train on.
            k (int): The number of clusters to group the data into.
            threshold (float): The threshold to stop convergence.
            maxEpochs (int): The maximum number of epochs to train for.

        Returns:
            tuple: A tuple of the clusters, the unassigned vectors, and the number of epochs trained for.
        """

        maxDist = Calc_Max_Possible_Distance(matrix)
        startingMinDist: float = maxDist * threshold
        isConverged: bool = False
        epoch = 0
        
        # Create k empty clusters
        clusters: list[Cluster] = [
            Cluster(
                name=f"cluster_{i}",
                centroid=choice(matrix), # Randomly select a vector from the matrix
            ) for i in range(self.kGroups)
        ]

        # Add a cluster to store unassigned vectors
        clusters.append(Cluster(
            name="unassigned",
            centroid=choice(matrix), # Randomly select a vector from the matrix
        ))

        while not isConverged:
            # Clear the vectors in each cluster
            for clusterObject in clusters:
                clusterObject.vectors = []

            # Assign each vector in the matrix to its closest centroid
            for vector in matrix:
                closestCluster: Cluster = clusters[-1]  # Initialize to the unassgined cluster
                closestClusterDistance: float = startingMinDist

                for clusterObject in clusters:
                    # Calculate the distance between the vector and the centroid
                    currentDistance: float = Calc_Distance(
                        aVector=vector,
                        bVector=clusterObject.centroid,
                    )

                    # Conduct comparison
                    if currentDistance < closestClusterDistance:
                        closestCluster = clusterObject
                        closestClusterDistance = currentDistance

                # Add the vector to the closest cluster
                closestCluster.vectors.append(vector)

            # Recalculate the centroid of each cluster
            for clusterObject in clusters:
                oldCentroid: list[int|float] = clusterObject.centroid
                newCentroid: list[int|float] = clusterObject.Recalculate_Centroid()

                if oldCentroid != newCentroid:
                    clusterObject.centroid = newCentroid
                    isConverged = False
                else:
                    isConverged = True

            epoch += 1
            if epoch > maxEpochs:
                break

        # Seperate the unassigned vectors from the clusters
        unassignedVectors: list[list[int|float]] = clusters[-1].vectors
        clusters = clusters[:-1]

        return clusters, unassignedVectors, epoch

def Calc_Distance(aVector: list[int|float], bVector: list[int|float]) -> float:
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
        aVector (list[int|float]): The first vector.
        bVector (list[int|float]): The second vector.

    Returns:
        float: The Euclidean distance between the two vectors.

    Raises:
        ValueError: If the vectors do not have the same dimensionality.
    """
    
    if len(aVector) != len(bVector):
        raise ValueError(f"Vectors must have the same dimensionality: {len(aVector)} != {len(bVector)}")

    distance: float = 0.0
    preRootSum: float = 0.0

    # Sum of squared differences
    for i in range(len(aVector)):
        preRootSum += (aVector[i] - bVector[i])**2

    # Square root of the sum
    distance = preRootSum**0.5

    return distance
    
def Calc_Max_Possible_Distance(matrix: list[list[int|float]]) -> float:
    """
    Calculate the maximum possible Euclidean distance in a given matrix.

    Parameters:
        matrix (list[list[int|float]]): The matrix to calculate the maximum distance for.

    Returns:
        float: The maximum possible Euclidean distance in the matrix.
    """

    dimensionCount: int = len(matrix[0])
    maxScalar: float|int = 0.0
    minScalar: float|int = 0.0

    # Find the maximum scalar in the matrix
    for vector in matrix:
        currentMax: float|int = max(vector)
        if currentMax > maxScalar:
            maxScalar = currentMax

    # Find the minimum scalar in the matrix
    for vector in matrix:
        currentMin: float|int = min(vector)
        if currentMin < minScalar:
            minScalar = currentMin

    return (maxScalar - minScalar) * (dimensionCount**0.5)