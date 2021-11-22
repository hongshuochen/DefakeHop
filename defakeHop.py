import os
import shutil
import pickle
import numpy as np
import multiprocessing
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import multiprocessing
from multi_cwSaab import MultiChannelWiseSaab

class DefakeHop():
    def __init__(self, num_hop=3, kernel_sizes=[3,3,3], split_thr=0.01, keep_thr=0.001, max_channels=[10,10,10], spatial_components=[0.95,0.95,0.95], n_jobs=4, verbose=True):
        self.num_hop = num_hop
        self.kernel_sizes = kernel_sizes
        self.split_thr = split_thr
        self.keep_thr = keep_thr
        self.max_channels = max_channels
        self.spatial_components = spatial_components
        self.multi_cwSaab = None
        self.spatial_PCA = {}
        self.channel_wise_clf = {}
        self.features = {}
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, images, labels):
        if self.verbose:
            print("===============DefakeHop Training===============")
        self.fit_multi_cwSaab(images)
        saab_features = self.transform_multi_cwSaab(images)
        del images
        # feature distillation
        # part1: spatial dimension reduction
        if self.verbose:
            print("===============Spatial Dimension Reduction===============")
        for hop in range(1, self.num_hop+1):
            self.features[hop] = {}
            features = saab_features['Hop' + str(hop)]
            if len(features) > self.max_channels[hop-1]:
                features = features[:,:,:,:self.max_channels[hop-1]]
            self.fit_spatial_PCA(features, hop)
            if self.verbose:
                print("Input shape:", features.shape[1:3], features.shape[1]*features.shape[2])
            for channel in range(features.shape[-1]):
                channel_wise_features = features[:,:,:,channel]
                channel_wise_features = self.transform_spatial_PCA(channel_wise_features, hop)
                self.features[hop][channel] = channel_wise_features
                if channel == 0 and self.verbose:
                    print("Output shape:", channel_wise_features.shape[-1])
        del saab_features
        # part2: semantic meaning distillation (soft classifiers)
        if self.verbose:
            print("===============Soft Classifiers===============")
        fit_all_channel_wise_clf(self.features, labels, n_jobs=4)
        self.set_all_channel_wise_clf()
        features = self.predict_all_channel_wise_clf(self.features)
        if self.verbose:
            print("Output shape:", features.shape)
        self.features = {}
        return features

    def predict(self, images):
        if self.verbose:
            print("===============DefakeHop Prediction===============")
        saab_features = self.transform_multi_cwSaab(images)
        del images
        # feature distillation
        # part1: spatial dimension reduction
        if self.verbose:
            print("===============Spatial Dimension Reduction===============")
        for hop in range(1, self.num_hop+1):
            self.features[hop] = {}
            features = saab_features['Hop' + str(hop)]
            if len(features) > self.max_channels[hop-1]:
                features = features[:,:,:,:self.max_channels[hop-1]]
            if self.verbose:
                print("Input shape:", features.shape[1:3], features.shape[1]*features.shape[2])
            for channel in range(features.shape[-1]):
                channel_wise_features = features[:,:,:,channel].reshape(features.shape[0], -1)
                channel_wise_features = self.transform_spatial_PCA(channel_wise_features, hop)
                self.features[hop][channel] = channel_wise_features
                if channel == 0 and self.verbose:
                    print("Output shape:", channel_wise_features.shape[-1])
        del saab_features
        # part2: semantic meaning distillation (soft classifiers)
        if self.verbose:
            print("===============Soft Classifiers===============")
        features = self.predict_all_channel_wise_clf(self.features)
        if self.verbose:
            print("Output shape:", features.shape)
        self.features = {}
        return features

    
    def fit_multi_cwSaab(self, images):
        # extract features
        multi_cwSaab = MultiChannelWiseSaab(num_hop=self.num_hop, kernel_sizes=self.kernel_sizes, 
                                            split_thr=self.split_thr, keep_thr=self.keep_thr)
        multi_cwSaab.fit(images, verbose=self.verbose)
        self.multi_cwSaab = multi_cwSaab
    
    def transform_multi_cwSaab(self, images):
        return self.multi_cwSaab.transform(images, verbose=self.verbose )

    def fit_spatial_PCA(self, features, hop):
        # train spatial pca for each hop
        pca = PCA(n_components=self.spatial_components[hop-1], svd_solver='full')
        # move the last axis to the second axis and flatten
        features = np.moveaxis(features, -1, 1)
        features = features.reshape(features.shape[0]*features.shape[1], -1)
        pca.fit(features)
        self.spatial_PCA[hop] = pca
    
    def transform_spatial_PCA(self, features, hop):
        # transform channel-wise data
        # flatten
        features = features.reshape(features.shape[0], -1)
        # spatial pca transformation
        pca = self.spatial_PCA[hop]
        return pca.transform(features)
    
    def set_all_channel_wise_clf(self):
        for hop in range(1, self.num_hop+1):
            self.channel_wise_clf[hop] = {}
            for channel in range(len(self.features[hop])):
                clf = pickle.load(open("tmp/" + str(hop) + '/' + str(channel) + ".pkl", "rb" ))
                self.channel_wise_clf[hop][channel] = clf
        shutil.rmtree("tmp")

    def predict_all_channel_wise_clf(self, features):
        prob = []
        for hop in range(1, self.num_hop+1):
            for channel in range(len(self.features[hop])):
                cw_prob = self.predict_channel_wise_clf(self.features[hop][channel], hop, channel)
                prob.append(cw_prob)
        prob = np.array(prob)
        return prob.T

    def predict_channel_wise_clf(self, features, hop, channel):
        clf = self.channel_wise_clf[hop][channel]
        return clf.predict_proba(features)[:,1]

def fit_all_channel_wise_clf(features, labels, n_jobs=4):
    parameters = []
    for hop in range(1,len(features)+1):
        for channel in range(len(features[hop])):
            parameters.append([features[hop][channel], labels, hop, channel])
    
    pool = multiprocessing.Pool(1)
    pool.starmap(fit_channel_wise_clf, parameters)

def fit_channel_wise_clf(features, labels, hop, channel):
    print("===Hop", hop, "Channel",  channel, "Start===")
    labels = labels.astype(int)
    clf = XGBClassifier(max_depth=1, tree_method='gpu_hist', objective='binary:logistic', eval_metric='auc', 
                        scale_pos_weight=(len(labels[labels==0])/len(labels[labels==1])), 
                        use_label_encoder=False)
    clf.fit(features, labels)
    if not os.path.exists("tmp/" + str(hop) + '/' + str(channel)):
        os.makedirs("tmp/" + str(hop) + '/' + str(channel))
    pickle.dump(clf, open("tmp/" + str(hop) + '/' + str(channel) + ".pkl", "wb" ))
    print("===Hop", hop, "Channel",  channel, "Finish===")

if __name__ == '__main__':
    import time
    from sklearn.datasets import fetch_olivetti_faces
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True)
    data = faces.reshape(-1,64,64,1)
    labels = np.ones(len(data))
    labels[:int(len(labels)/2)] = 0
    defakehop = DefakeHop()
    prob1 = defakehop.fit(data, labels) 
    prob2 = defakehop.predict(data)
    print(np.sum(np.abs(prob1-prob2)))