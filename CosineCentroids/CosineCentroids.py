import random

class Cluster:
    def __init__(self, name: str, centroid: list[int|float] = []):
        self.name = name
        self.centroid = centroid
        self.vectors: list[list[int|float]] = []

    def _Average_Vector(self, vectors: list[list[int|float]]) -> list[int|float]:
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
        if self.vectors == []: return True

        previousCentroid: list[int|float] = self.centroid
        self.centroid = self._Average_Vector(self.vectors)

        return previousCentroid == self.centroid
    
    def To_Dict(self) -> dict[str, str | list[int|float] | list[list[int|float]]]:
        return {
            "name": self.name,
            "centroid": self.centroid,
            "vectors": self.vectors
        }

class Cosine_Centroids:
    def __init__(self,
            trainingMatrix:list[list[int|float]], 
            nGroups: int = 3,
            maxEpochs: int = 100
        ) -> None:

        self.trainingMatrix: list[list[int|float]] = trainingMatrix
        self.nGroups: int = nGroups
        self.maxEpochs: int = maxEpochs

        self.clusters: list[Cluster] = []
        self.epoch: int = 0

        # Auto train
        if self.trainingMatrix:
            self.Train()

    def _Get_Dimensionality(self) -> int:
        return len(self.trainingMatrix[0])
    
    def _Get_Random_Vectors(self, count: int = 1) -> list[list[int|float]]:
        return random.choices(self.trainingMatrix, k=count)

    def _Initialize_Clusters(self, count: int) -> list[Cluster]:
        clusters: list[Cluster] = []
        randomVectors: list[list[int|float]] = self._Get_Random_Vectors(count=count)

        for i in range(count):
            clusters.append(
                Cluster(
                    name=f"cluster_{i}",
                    centroid=randomVectors[i]
                )
            )

        return clusters
    
    def _Cosine_Similarity(self, aVector: list[int|float], bVector: list[int|float]) -> float:
        if len(aVector) != len(bVector):
            raise ValueError(f"Vectors must have the same dimensionality: {len(aVector)} != {len(bVector)}")

        dotProduct: float = sum(a * b for a, b in zip(aVector, bVector))
        magnitudeA: float = sum(a * a for a in aVector) ** 0.5
        magnitudeB: float = sum(b * b for b in bVector) ** 0.5

        return dotProduct / (magnitudeA * magnitudeB)
        
    def Train(self) -> None:
        # Initialize clusters
        self.clusters = self._Initialize_Clusters(count=self.nGroups)

        # Train model
        isConverged: bool = False
        self.epoch = 0

        while not isConverged:
            # Clear the clusters
            for cluster in self.clusters:
                cluster.vectors = []

            # Assign vectors to clusters
            for vector in self.trainingMatrix:
                # Calculate cosine similarities
                cosineSimilarities: dict[int, float] = {
                    centroid: self._Cosine_Similarity(vector, cluster.centroid) for centroid, cluster in enumerate(self.clusters)
                }

                # Get the closest centroid
                closestCentroid: int = max(cosineSimilarities, key=cosineSimilarities.get)

                # Add the vector to the closest centroid
                self.clusters[closestCentroid].vectors.append(vector)

            # Calculate new centroids
            convergenceArray: list[bool] = [
                cluster.Recalculate_Centroid() for cluster in self.clusters
            ]

            # Check for convergence
            isConverged = all(convergenceArray)
            self.epoch += 1

            # Check for maximum epochs
            if self.epoch >= self.maxEpochs:
                break   

    def Predict(self, vector: list[int|float]) -> Cluster:
        cosineSimilarities: dict[int, float] = {
            centroid: self._Cosine_Similarity(vector, cluster.centroid) for centroid, cluster in enumerate(self.clusters)
        }
        clusterIndex: int = max(cosineSimilarities, key=cosineSimilarities.get)
        return self.clusters[clusterIndex]
    