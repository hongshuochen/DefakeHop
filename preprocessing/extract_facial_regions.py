import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from PIL import Image
from face_aligner import FaceAligner
from face_aligner import FACIAL_LANDMARKS_68_IDXS

def save(img, filename):
    Image.fromarray(img).save(filename)

def extract_patch(img, point, patch_size):
    patch = img[int(point[1]-patch_size/2): int(point[1]+patch_size/2),
                int(point[0]-patch_size/2): int(point[0]+patch_size/2)] 
    return patch

def PatchExtraction(video_path, landmarks_path, output_dir, patch_size=32):
    print("Input: ", video_path)
    print("Output:", output_dir)
    frames = []
    frame_number = []
    if os.path.exists(landmarks_path) == False:
        return
    df = pd.read_csv(landmarks_path)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # if count % 6 == 0 and df[' success'][count] == 1:
        # if df[' success'][count] == 1:
        if count % 6 == 0 and len(df['success']) > count:
            if df['success'][count] == 1:
                frame = frame[:,:,::-1]
                frames.append(frame)
                frame_number.append(count)
        count += 1
    cap.release()

    folders = ["aligned_face", "left_eye", "right_eye", "mouth", "nose"]
    for folder in folders:
        directory = os.path.join(output_dir, folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

    for idx, frame in enumerate(frames):
        x = np.array(df.iloc[frame_number[idx],299:299+68]).reshape(68,-1)
        y = np.array(df.iloc[frame_number[idx],299+68:299+68*2]).reshape(68,-1)
        z = np.ones(68).reshape(68,-1)
        landmarks = np.concatenate((x,y), axis=1)
        aligner = FaceAligner(desiredLeftEye=(0.35, 0.35), desiredFaceWidth=128, desiredFaceHeight=int(128*2))
        aligned_face, M = aligner.align(frame, landmarks)

        landmarks_z = np.concatenate((landmarks, z), axis=1)
        affined_landmarks = np.matmul(landmarks_z, M.transpose())

        regions = ["left_eye", "right_eye", "mouth", "nose"]
        regions_image = []
        for region in regions:
            start, end = FACIAL_LANDMARKS_68_IDXS[region]
            Pts = affined_landmarks[start:end]
            Center = Pts.mean(axis=0)
            try:
                img = extract_patch(aligned_face, Center, patch_size)
            except:
                break
            if img.shape != (32, 32, 3):
                break
            regions_image.append(img)
        
        if len(regions_image) == len(regions):
            for i, region in enumerate(regions):
                filename = os.path.join(output_dir, region, str(frame_number[idx]).zfill(4) + '.bmp')
                img = regions_image[i]
                save(img, filename)
            filename = os.path.join(output_dir, 'aligned_face', str(frame_number[idx]).zfill(4)  + '.bmp')
            np.save(os.path.join(output_dir, 'aligned_face', str(frame_number[idx]).zfill(4)  + '.npy'), affined_landmarks)
            save(aligned_face, filename)


if __name__ == "__main__":
    '''
    Extract Patches from Landmarks
    videos_path: path to the video folder
    landmarks_path: path to landmarks folder
    output_dir: path to output directory
    '''
    videos_path = os.path.abspath(sys.argv[1])
    landmarks_path = os.path.abspath(sys.argv[2])
    output_dir = os.path.abspath(sys.argv[3])
    print(videos_path)
    print(landmarks_path)
    print(output_dir)
    
    videos = os.listdir(videos_path)
    videos = [vid for vid in videos if '.mp4' in vid]
    parameters = []
    for video in videos:
        video_path = os.path.join(videos_path, video)
        video = video.replace('.mp4', '')
        csv_path = os.path.join(landmarks_path, video + '.csv')
        output_path = os.path.join(output_dir, video)
        parameters.append([video_path, csv_path, output_path])
    pool = multiprocessing.Pool()
    pool.starmap(PatchExtraction, parameters)
