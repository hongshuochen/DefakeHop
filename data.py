import os
import cv2
import numpy as np

def load(real_path, fake_path, region = 'left_eye'):
    images = []
    labels = []
    names = []

    for vid_id in os.listdir(real_path):
        vid_id = vid_id.replace('.mp4', '')
        patches_path = os.path.join("patches", real_path, vid_id, region)
        patches = os.listdir(patches_path)
        patches.sort()
        for patch in patches:
            img = cv2.imread(os.path.join(patches_path, patch))[:,:,::-1]
            images.append(img)
            labels.append(0)
            names.append('real/' +  vid_id + '_' + patch)
    
    for vid_id in os.listdir(fake_path):
        vid_id = vid_id.replace('.mp4', '')
        patches_path = os.path.join("patches", fake_path, vid_id, region)
        patches = os.listdir(patches_path)
        patches.sort()
        for patch in patches:
            img = cv2.imread(os.path.join(patches_path, patch))[:,:,::-1]
            images.append(img)
            labels.append(1)
            names.append('fake/' +  vid_id + '_' + patch)


    return np.array(images), np.array(labels), np.array(names)

if __name__ == '__main__':
    for split in ["train", "test"]:
        for region in ["left_eye", "right_eye", "mouth"]:
            print(region)
            images, labels, names = load(split + "/real", split + "/fake", region=region)
            np.savez("data/" + region + '.' + split + '.npz', images=images, labels=labels, names=names)

            # print("Training Videos: ", len(set([vid_name(name) for name in train_names])))
            # print("Testing Videos: ", len(set([vid_name(name) for name in test_names])))
            # print("Training Real Videos: ", len(set([vid_name(name) for i, name in enumerate(train_names) if train_labels[i] == 1])))
            # print("Training Fake Videos: ", len(set([vid_name(name) for i, name in enumerate(train_names) if train_labels[i] == 0])))
            # print("Testing Real Videos: ", len(set([vid_name(name) for i, name in enumerate(test_names) if test_labels[i] == 1])))
            # print("Testing Fake Videos: ", len(set([vid_name(name) for i, name in enumerate(test_names) if test_labels[i] == 0])))
            # print("Training Image Shape: ", train_images.shape)
            # print("Testing Image Shape: ", test_images.shape)
