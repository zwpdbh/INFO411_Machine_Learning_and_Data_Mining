
# business as usual
import numpy as np
import matplotlib.pylab as pl

# your code (function definition, plus testing) here

# Because the condition of converges is comparing the cos(theta) of two vectors. So in some rare situation, it may exceed the maximum depth of recursion.
# In that situation, just re-run the code, it will be ok.
class MYPCA:
    def __init__(self, data):
        self.X = np.matrix(data)
        self.num_of_instances = self.X.shape[0]
        self.dim_of_data = self.X.shape[1]

        # make it zero meaned:
        self.m = np.mean(self.X, axis=0)
        self.X -= self.m

        # create covariance matrix
        self.covariance_matrix = (self.X.T * self.X) / self.num_of_instances

        self.eigens = []

    def get_covariance_matrix(self):
        return self.covariance_matrix

    def findOneEigenvectorWithInitialVector(self, w):
        e = self.covariance_matrix * w
        e = e / np.sqrt(e.T * e)

        if np.dot(e.T, w) == 1:
            return e
        else:
            return self.findOneEigenvectorWithInitialVector(e)

    def findEigenVectors(self):
        w = np.matrix(np.ones(self.dim_of_data)).T
        w = w / np.sqrt(w.T * w)
        e1 = self.findOneEigenvectorWithInitialVector(w)
        self.eigens.append(e1)

        #         self.covariance_matrix = self.covariance_matrix - self.covariance_matrix * e1 * e1.T
        #         w = np.matrix(np.ones(self.dim_of_data)).T
        #         w = w / np.sqrt(w.T * w)
        #         e2 = self.findOneEigenvectorWithInitialVector(w)

        #         self.eigens.append(e2)

        return np.hstack(self.eigens)


bank = np.loadtxt('bank_new.txt')
bank = np.random.permutation(bank)



# label = bank[:, -1]
# sub0 = label[:] == 0
# sub1 = label[:] == 1
#
pca = MYPCA(data=bank)
eigenMatrix = pca.findEigenVectors()
print eigenMatrix.shape
# print eigenMatrix
print bank.shape, type(bank)
print eigenMatrix.shape, type(eigenMatrix)

projectedData = bank * eigenMatrix
print projectedData.shape, type(projectedData)

# pl.plot(projectedData[sub0,0],projectedData[sub0,1],'bx')
# pl.plot(projectedData[sub1,0],projectedData[sub1,1],'g+')
# print projectedData