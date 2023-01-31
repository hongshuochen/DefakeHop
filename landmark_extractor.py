import os
import multiprocessing

OPENFACE_EXE_DIR = "/home/techsigncloud/ExtLibs/OpenFace/build/bin"

def LandmarkExtractor(file_path, output_dir):
    if os.path.exists(output_dir):
        print(file_path, " already exists, skipping")
        return False
    print("Input: ", file_path)
    stream = os.popen(f"{OPENFACE_EXE_DIR}/FeatureExtraction -f {file_path} -2Dfp -out_dir {output_dir}")
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