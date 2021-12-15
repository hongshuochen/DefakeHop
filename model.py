import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import utils
from defakeHop import DefakeHop

class Ensemble():
    def __init__(self, regions=['left_eye', 'right_eye', 'mouth'], num_frames=6, verbose=True):
        self.regions = regions
        self.num_frames = num_frames
        self.defakeHop = {}
        self.classifier = None
        self.features = {}
        self.labels = None
        self.names = None
        self.predict_features = {}
        self.predict_names = None
        self.verbose = verbose

    def fit_region(self, region, images, labels, names, multi_cwSaab_parm):
        # feed the data of a region to generate features for training
        # calling this function several times for differnet regions
        if self.verbose:
            print("==============================" + region + "==============================")
        if len(self.defakeHop) == 0:
            self.labels = labels
            self.names = names
        else:
            assert np.array_equal(names, self.names), "the order of names are not aligned with previous regions"
            assert np.array_equal(labels, self.labels), "the order of labels are not aligned with previous regions"
        defakehop = DefakeHop(**multi_cwSaab_parm)
        features = defakehop.fit(images, labels)
        self.defakeHop[region] = defakehop
        self.features[region] = features
        return self

    def predict_region(self, region, images, names):
        # feed the data of a region to generate features for prediction
        # calling this function several times for different regions
        if self.verbose:
            print("==============================" + region + "==============================")
        if self.predict_names is None:
            self.predict_names = names
        else:
            assert np.array_equal(names, self.predict_names), "the order of names are not aligned with previous regions"
        self.predict_features[region] = self.defakeHop[region].predict(images)
    
    def clean_buffer(self):
        # clean the buffer for all regions
        self.features = {}
        self.labels = None
        self.names = None
        self.predict_features = {}
        self.predict_names = None
        
    def concatenate_features(self, regions=None, train_flag=True):
        # concatenate regions and adjacent frames features
        if self.verbose:
            print("===============Concatenation===============")
        if regions is None:
            regions = self.regions
        features = self.concatenate_regions_features(regions,train_flag=train_flag)
        if train_flag:
            names = self.names
        else:
            names = self.predict_names
        return self.concatenate_frames_features(features, self.labels, names, train_flag=train_flag)

    def concatenate_regions_features(self, regions, train_flag):
        # concatnate all regions features
        features = []
        for region in regions:
            if train_flag:
                assert region in self.features, 'Please train ' + region + ' first'
                features.extend(self.features[region].T.tolist())
            else:
                assert region in self.predict_features, 'Please train ' + region + ' first'
                features.extend(self.predict_features[region].T.tolist())
        return np.array(features).T

    def concatenate_frames_features(self, features, labels, names, train_flag):
        # concatenate adjacent frames features
        frames_labels = []
        frames_names = []
        all_frames_features = []
    
        for idx, prob in enumerate(features):
            cur_vid_name = utils.vid_name(names[idx])
            cur_frame = utils.frame(names[idx])
            frames_features=[]
            for i in range(self.num_frames):
                if idx+i < len(names) and utils.vid_name(names[idx+i]) == cur_vid_name and utils.frame(names[idx+i]) == cur_frame + i*6:
                    frames_features.extend(features[idx+i])
                else:
                    break
            if len(frames_features) == self.num_frames*len(prob):
                all_frames_features.append(frames_features)
                if train_flag:
                    frames_labels.append(labels[idx])
                frames_names.append(names[idx])
        if train_flag:
            frames_labels = np.array(frames_labels)
        frames_names = np.array(frames_names)
        all_frames_features = np.array(all_frames_features)
        return all_frames_features, frames_labels, frames_names

    def train_classifier(self, folds=4, param_comb=20, clean=True):
        # train the final classifier to decide whether the face real or fake
        features, labels, names = self.concatenate_features()
        if self.verbose:
            print("===============Training Classifier===============")
            print("Features shape:", features.shape)
        params = {
        'min_child_weight': [1, 2, 3, 5, 7, 11, 13, 17, 19, 23],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.6, 0.8, 1],
        'max_depth': [6]
        }
        labels = labels.astype(int)
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.2, eval_metric='auc', objective='binary:logistic',
                            tree_method='gpu_hist', scale_pos_weight=(len(labels[labels==0])/len(labels[labels==1])),
                            use_label_encoder=False)
        skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 0)
        clf = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=8, cv=skf.split(features,labels), random_state=1001)
        clf.fit(features, labels)
        self.classifier = clf.best_estimator_
        print(self.classifier)
        prob = clf.predict_proba(features)[:,1]
        if clean:
            self.clean_buffer()
        return prob, names
    def predict_classifier(self, clean=True):
        # decide whether the face real or fake
        features, _, names = self.concatenate_features(train_flag=False)
        if self.verbose:
            print("===============Prediction===============")
            print("Features shape:", features.shape)
        prob = self.classifier.predict_proba(features)[:,1]
        if clean:
            self.clean_buffer()
        return prob, names

if __name__ == '__main__':
    # regions: left_eye, right_eye, mouth, ....
    # num_frames: a positive integer value
    def vid_name(name):
        return name.replace(name.split('_')[-1],'')[0:-1]
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    model = Ensemble(regions=['left_eye', 'right_eye', 'mouth'], num_frames=6, verbose=True)
    # more parameters for multi channel-wise Saab feature extraction
    multi_cwSaab_parm = dict(num_hop=3, kernel_sizes=[3,3,3], split_thr=0.01, keep_thr=0.001, 
                            max_channels=[10,10,10], spatial_components=[0.9,0.9,0.9], n_jobs=4, verbose=True)
    for region in model.regions:
        path = 'data/' + region + '.train.npz'
        data = np.load(path)
        train_images = data['images']
        train_labels = data['labels']
        train_names = data['names']
        print(train_images.shape)
        model.fit_region(region, train_images, train_labels, train_names, multi_cwSaab_parm)
    train_prob, train_vid_names = model.train_classifier()
    for region in model.regions:
        path = 'data/' + region + '.test.npz'
        data = np.load(path)
        test_labels = data['labels']
        test_images = data['images']
        test_names = data['names']
        model.predict_region(region, test_images, test_names)

    prob, names = model.predict_classifier()
    print("===============Training Results===============")
    utils.evaluate(train_prob, train_vid_names)
    print("===============Testing Results===============")
    vid_gts , vid_probs, vid_names = utils.evaluate(prob, names)
    import pandas as pd
    df = pd.DataFrame({'name':vid_names, 'label':vid_gts, 'prob':vid_probs})
    df.to_csv('output.csv', index=False)
