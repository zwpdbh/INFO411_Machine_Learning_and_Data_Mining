import numpy as np


class MySelfOrganizingMap:
    def __init__(self, size, data, iterations, initialNeighborRadius, learningRate):
        self.iterations = iterations
        self.updateRadius = initialNeighborRadius
        self.gamma = learningRate
        self.mapSize = size
        self.data = data
        self.dataSize = data.shape[0]

        # random select data vector for initial vectors in map
        w = data[np.random.randint(self.dataSize)].copy()
        for i in range(self.mapSize * self.mapSize - 1):
            w = np.vstack((w, data[np.random.randint(self.dataSize)].copy()))

        self.nodes = w.reshape(self.mapSize, self.mapSize, -1).astype(float)

    def findWinner(self, w):
        winner = np.argmin(np.linalg.norm(self.nodes - w, axis=2))
        return winner / self.mapSize, winner % self.mapSize

    def updateNodes(self, x, i, j, radius, gamma):
        for m in range(-radius, radius + 1):
            for n in range(-radius, radius + 1):
                # check the boundary
                if (i + m) < 0 or (i + m) >= self.mapSize: continue
                if (j + n) < 0 or (j + n) >= self.mapSize: continue
                self.nodes[i + m, j + n] += gamma * (x - self.nodes[i + m, j + n])

    def train(self):
        for t in range(self.iterations):
            x = self.data[np.random.randint(self.dataSize)]
            i, j = self.findWinner(x)
            self.updateNodes(x, i, j,
                             radius=self.updateRadius * (self.iterations - t) / self.iterations,
                             gamma=self.gamma * (self.iterations - t) / self.iterations
                             )
