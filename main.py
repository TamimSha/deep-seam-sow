from utils import generateDataset
import os

path = "data/videos/"
videofiles = os.listdir(path)

totalfiles = 0
for i in range(0, len(videofiles)):
    print(f"File {i+1} of {len(videofiles)}\n")
    totalfiles += generateDataset(path + videofiles[i], 4*2048, 4, "data/datasets/dataset_02/",
                    128, start_clip=1145, save_index=totalfiles)