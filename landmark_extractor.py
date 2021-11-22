import os
import multiprocessing

def LandmarkExtractor(file_path, output_dir):
    print("Input: ", file_path)
    stream = os.popen("OpenFace/build/bin/FeatureExtraction -f '{input}' -out_dir '{output}'"
            .format(input = file_path, output = output_dir))
    output = stream.read()
    print("Output:", output_dir)
    return True

if __name__ == "__main__":
    paths = ["train/fake", "train/real", "test/real", "test/fake"]
    params = []
    for path in paths:
        for vid in os.listdir(path):
            if '.mp4' in vid:
                params.append([os.path.join(path, vid), os.path.join("landmarks", path, vid.replace(".mp4", ""))])

    pool = multiprocessing.Pool(8)
    pool.starmap(LandmarkExtractor, params)