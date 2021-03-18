import time
import numpy as np
from skimage.measure import block_reduce

from saab import Saab

class MultiChannelWiseSaab():
    def __init__(self, num_hop=3, kernel_sizes=[3,3,3], split_thr=0.01, keep_thr=0.001):
        self.num_hop = num_hop
        self.kernel_sizes = kernel_sizes
        self.split_thr = split_thr
        self.keep_thr = keep_thr
        self.saabs = {}
        self.energies = {}
        self.features = {}
        self.info = {}
        self.tmp = []

    def fit(self, images, verbose=False):
        start = time.time()
        if verbose:
            print("===============MultiChannelWiseSaab Training===============")
        self.fit_hop1(images, verbose=verbose)
        for n in range(2,self.num_hop+1):
            self.fit_hop_n(n, verbose=verbose)
        self.features = {}
        if verbose:
            print("spent", time.time()-start, "s")
        return self

    def transform(self, images, verbose=False):
        start = time.time()
        if verbose:
            print("===============MultiChannelWiseSaab Transformation===============")
        self.transform_hop1(images, verbose=verbose)
        for n in range(2,self.num_hop+1):
            self.transform_hop_n(n, verbose=verbose)
        if verbose:
            print("spent", time.time()-start, "s")
            print("===============Features Dimensions===============")
        for n in range(1,self.num_hop+1):
            if len(self.features['Hop' + str(n)]) != 0:
                self.features['Hop' + str(n)] = np.concatenate(self.features['Hop' + str(n)], axis=3)
                print('Hop' + str(n), self.features['Hop' + str(n)].shape)
        features = self.features
        self.features = {}
        return features
        
    def max_pooling(self, images):
        return block_reduce(images, (1,2,2,1), np.max)

    def fit_hop1(self, images, verbose):
        # train the first hop
        if verbose:
            print('Hop1')
            print("Input shape:", images.shape)
        saab = Saab(kernel_size=self.kernel_sizes[0], bias_flag=False)
        saab.fit(images)
        self.saabs['Hop1'] = [saab]
        self.energies['Hop1'] = [saab.eigenvalues/sum(saab.eigenvalues)]
        n_channels = np.sum(self.energies['Hop1'][0] > self.keep_thr)
        output = saab.transform(images, n_channels)
        self.features['Hop1'] = [self.max_pooling(output)]
        self.info['Hop1'] = [(0, 0, n_channels)]
        if verbose:
            print("Output shape:", self.features['Hop1'][-1].shape)
    
    def transform_hop1(self, images, verbose):
        # transform the first hop
        if verbose:
            print('Hop1')
            print("Input shape:", images.shape)
        saab = self.saabs['Hop1'][0]
        n_channels = self.info['Hop1'][0][2]
        output = saab.transform(images, n_channels)
        self.features['Hop1'] = [self.max_pooling(output)]
        if verbose:
            print("Output shape:", self.features['Hop1'][-1].shape)
        
    
    def fit_hop_n(self, n, verbose):
        # train the nth hop (n > 1)
        if verbose:
            print('Hop' + str(n))
        self.saabs['Hop' + str(n)] = []
        self.energies['Hop' + str(n)] = []
        self.features['Hop' + str(n)] = []
        self.info['Hop' + str(n)] = []
        for saab_id in range(len(self.saabs['Hop' + str(n-1)])):
            saab_parent = self.saabs['Hop' + str(n-1)][saab_id]
            energies_parent = self.energies['Hop' + str(n-1)][saab_id]
            features_parent = self.features['Hop' + str(n-1)][saab_id]
            for channel_id in range(len(energies_parent)):
                energy = energies_parent[channel_id]
                if energy > self.split_thr:
                    features = features_parent[:,:,:,channel_id][...,np.newaxis]
                    if verbose:
                        print("SaabID:", saab_id, "ChannelID:", channel_id, "Energy:", energy)
                        print("Input shape:", features.shape)
                    saab = Saab(kernel_size=self.kernel_sizes[n-1], bias_flag=True)
                    saab.fit(features)
                    self.saabs['Hop' + str(n)].append(saab)
                    energies = saab.eigenvalues/sum(saab.eigenvalues)*energy
                    self.energies['Hop' + str(n)].append(energies)
                    n_channels = np.sum(energies> self.keep_thr)
                    output = saab.transform(features, n_channels)
                    self.tmp.append((saab_id,channel_id,self.max_pooling(output)))
                    self.features['Hop' + str(n)].append(self.max_pooling(output))
                    self.info['Hop' + str(n)].append((saab_id, channel_id, n_channels))
                    if verbose:
                        print("Output shape:", self.features['Hop' + str(n)][-1].shape)
                else:
                    break

    def transform_hop_n(self, n, verbose):
        # transfrom the nth hop (n > 1)
        if verbose:
            print('Hop' + str(n))
        self.features['Hop' + str(n)] = []
        index = 0
        for saab_id, channel_id, n_channels in self.info['Hop' + str(n)]:
            features_parent = self.features['Hop' + str(n-1)][saab_id]
            features = features_parent[:,:,:,channel_id][...,np.newaxis]
            if verbose:
                print("SaabID:", saab_id, "ChannelID:", channel_id)
                print("Input shape:", features.shape)
            saab = self.saabs['Hop' + str(n)][index]
            output = saab.transform(features, n_channels)
            self.features['Hop' + str(n)].append(self.max_pooling(output))
            if verbose:
                print("Output shape:", self.features['Hop' + str(n)][-1].shape)
            index += 1

if __name__ == '__main__':
    import time
    from sklearn.datasets import fetch_olivetti_faces
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True)
    data = faces.reshape(-1,64,64,1)

    # test MultiChannelWiseSaab
    multi_cwSaab = MultiChannelWiseSaab(num_hop=3, kernel_sizes=[3,3,3])
    multi_cwSaab.fit(data, verbose=True)
    output = multi_cwSaab.transform(data, verbose=True)


