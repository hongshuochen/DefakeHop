import numpy as np
from sklearn import metrics

def vid_name(name):
    return name.replace(name.split('_')[-1],'')[0:-1]

def frame(name):
    return int(name.split('_')[-1][:-4])

def vid_prob(probs, names):
    video = {}
    count = {}

    for name in names:
        video[vid_name(name)] = 0
        count[vid_name(name)] = 0
    for name in names:
        count[vid_name(name)] += 1
    for idx, prob in enumerate(probs):
        video[vid_name(names[idx])] += prob
    vid_gts = []
    vid_probs = []
    vid_names = []
    for key in video:
        if 'real' in key:
            vid_gts.append(0)
        else:
            vid_gts.append(1)
        video[key] = video[key]/count[key]
        vid_probs.append(video[key])
        vid_names.append(key)
    return vid_gts , vid_probs, vid_names

def evaluate(probs, names):
    labels = [int('real' not in i) for i in names]
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    roc_auc = metrics.auc(fpr, tpr)
    print("Frame AUC", roc_auc)
    vid_gts , vid_probs, vid_names = vid_prob(probs, names)
    fpr, tpr, thresholds = metrics.roc_curve(vid_gts, vid_probs)
    roc_auc = metrics.auc(fpr, tpr)
    print("Video AUC", roc_auc)
    # return fpr, tpr, roc_auc
    return vid_gts , vid_probs, vid_names