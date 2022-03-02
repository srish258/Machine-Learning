import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''
    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.
    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # TODO: implement the Kmeans++ initialization
    centers=[generator.randint(0,n)]
    for i in range(n_cluster-1):
        C_Dis= []
        for j in range(n):
            C_Dis.append(min([np.linalg.norm(x[j]-x[center])**2 for center in centers]))
        P_C = generator.rand()
        cdf = np.cumsum(C_Dis)/sum(C_Dis)
        cur_c = np.min(np.argwhere(cdf>P_C))
        centers.append(cur_c)
    ###############################################

    # DO NOT CHANGE CODE BELOW THIS LINE
    print(centers)
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

def getObjective(centroids,x,assignments,n_cluster):
    objective = np.sum([np.sum(np.linalg.norm(x[assignments==i]-centroids[i])**2) for i in range(n_cluster)])
    return objective


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0),
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        centroids = np.array([x[i] for i in self.centers])
        assert centroids.shape == (self.n_cluster,D)
        assignments = np.zeros(N)
        C_objective = np.sum([np.sum(np.linalg.norm(x[assignments==i]-centroids[i])**2) for i in range(self.n_cluster)])
        objective = C_objective/N

        update=0
        while update<self.max_iter:
            D2C = np.sum(np.square(x-np.expand_dims(centroids,axis=1)),axis=2)
            assignments = np.argmin(D2C,axis=0)
            C_objective = np.sum([np.sum(np.linalg.norm(x[assignments==i]-centroids[i])**2) for i in range(self.n_cluster)])
            cur_objective = C_objective/N
            if np.absolute(objective - cur_objective) < self.e:
                break
            objective = cur_objective
            curCentroids = np.array([np.mean(x[assignments == cluster_ind], axis=0) for cluster_ind in range(self.n_cluster)])
            curCentroids[np.where(np.isnan(curCentroids))] = centroids[np.where(np.isnan(curCentroids))]
            centroids = curCentroids
            update+=1

        ###################################################################


        assert centroids.shape == (self.n_cluster,D)
        assert assignments.shape == (N,)
        return (centroids,assignments,update)

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids,assignments,update = kmeans.fit(x,centroid_func)
        centroid_labels = []
        for i in range(self.n_cluster):
            labels = [y.item(i) for i in np.argwhere(assignments==i).reshape(-1)]
            centroid_labels.append(np.bincount(labels).argmax())
        centroid_labels = np.array(centroid_labels)
        ################################################################



        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        D2C = np.sum(np.square(x-np.expand_dims(self.centroids,axis=1)),axis=2)
        result = np.argmin(D2C,axis=0)
        ##########################################################################


        return np.array(self.centroid_labels[result])




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)
        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)
        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    N, M, C = image.shape
    data = image.reshape(N * M, C)
    r = np.argmin(np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2), axis=0)
    new_image = code_vectors[r].reshape(N, M, C)
    ##############################################################################

    return new_image
