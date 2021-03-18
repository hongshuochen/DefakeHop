import numpy as np
from numpy import linalg as LA
from skimage.util.shape import view_as_windows

class Saab():
    def __init__(self, kernel_size=3, bias_flag=False):
        self.bias = None
        self.features_mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.kernel_size = kernel_size
        self.bias_flag = bias_flag

    def PCA(self, X):
        # remove mean
        X = X - X.mean(axis=0)

        # calculate covariance matrix and calculate eigenvalues and eigenvectors of the covariance matrix
        # / (X.shape[0] - 1)) to make it consistent with the sklearn pca
        # self.eigenvalues, self.eigenvectors = LA.eig(np.matmul(X.transpose(), X)/ (X.shape[0] - 1))
        self.eigenvalues, self.eigenvectors = LA.eig(np.cov(X, rowvar=0))
        
        # absolute value of the eigenvalues
        self.eigenvalues = np.abs(self.eigenvalues)

        # sort from the largest to the smallest
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        return self
    
    def patch_extraction(self, images):
        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[-1]
        images = view_as_windows(images, (1, self.kernel_size, self.kernel_size, 1))
        return images.reshape(N, H, W, C * self.kernel_size**2)

    def fit(self, images, max_images=10000, max_patches=1000000, seed=777):
        images = np.array(images)
        images = images.astype('float64')

        # subsample images
        if len(images) > max_images:
            print("sampling images")
            np.random.seed(seed)
            images = images[np.random.choice(len(images),max_images, replace=False), :]
        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[3] * self.kernel_size**2

        # collect patches
        patches = self.patch_extraction(images)
        del images
        if len(patches) > max_patches:
            print("sampling patches")
            np.random.seed(seed)
            patches = patches[np.random.choice(len(patches), max_patches, replace=False), :]

        # flatten
        patches = patches.reshape(N*H*W, C)
        if self.bias_flag:
            self.bias=np.max(LA.norm(patches, axis=1))/np.sqrt(C)

        # remove mean
        self.features_mean = np.mean(patches,axis=0,keepdims=True)
        patches -= self.features_mean

        # remove patches mean
        patches_mean = np.mean(patches,axis=1,keepdims=True)
        patches -= patches_mean

        # dc_ev = np.var(patches_mean * np.sqrt(patches.shape[-1]))
        # dc_kernel =  np.ones((patches.shape[-1], 1))/np.sqrt(dc_ev)/np.sqrt(patches.shape[-1])
        
        # calculate eigenvectors and eigenvalues
        self.PCA(patches)

        # self.eigenvectors = np.concatenate((dc_kernel, self.eigenvectors[:,:-1]), axis=1)
        # self.eigenvalues = np.concatenate((np.array([dc_ev]), self.eigenvalues[:-1]), axis=0)
        return self
    
    def transform(self, images, n_channels=-1):
        images = np.array(images)
        images = images.astype('float64')

        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[3] * self.kernel_size**2

        # collect patches
        patches = self.patch_extraction(images)
        del images

        # flatten
        patches = patches.reshape(N*H*W, C)

        # remove mean
        patches -= self.features_mean

        # remove patches mean
        patches_mean = np.mean(patches,axis=1,keepdims=True)
        patches -= patches_mean
        

        if n_channels == -1:
            kernels = self.eigenvectors
            n_channels = len(self.eigenvalues)
        else:
            kernels = self.eigenvectors[:,:n_channels]

        if self.bias_flag:
            patches=patches+self.bias
            return np.matmul(patches, kernels).reshape(N, H, W, n_channels)
        else:
            return np.matmul(patches, kernels).reshape(N, H, W, n_channels)
                
if __name__ == '__main__':
    import time
    from sklearn.datasets import load_digits
    digits = load_digits()
    data = digits.data
    data = data.reshape(-1,8,8,1)

    # test Saab
    saab = Saab(bias_flag=True)
    start = time.time()
    saab.fit(data)
    print("training time:", time.time()-start, "s")

    start = time.time()
    output = saab.transform(data)
    print("transformation time:", time.time()-start, "s")

    # test PCA
    data = data.reshape(-1,64)
    from sklearn.decomposition import PCA
    pca = PCA()
    start = time.time()
    pca.fit(data)
    print("sklearn pca training time:", time.time()-start, "s")

    start = time.time()
    output = pca.transform(data)
    print("sklearn pca transform time:", time.time()-start, "s")

    saab = Saab()
    start = time.time()
    saab.PCA(data)
    print("numpy pca training time:", time.time()-start, "s")
    print("difference of eigenvalues:", np.sum(np.abs(pca.explained_variance_ - saab.eigenvalues)))
    assert np.sum(np.abs(pca.explained_variance_ - saab.eigenvalues)) <= 10**-10
    print("dot product of eigenvectors of sklearn pca and numpy pca:")
    print(np.diag(np.matmul(pca.components_, saab.eigenvectors)))