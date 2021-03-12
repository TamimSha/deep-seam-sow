from random import shuffle
from utils import makeDataset
from utils import DatasetGenerator
import matplotlib.pyplot as plt

# path = "data/videos/"
# videofiles = os.listdir(path).sort()

# totalfiles = 0
# for i in range(0, len(videofiles)):
#     print(f"File {i+1} of {len(videofiles)}\n")
#     totalfiles += makeDataset(path + videofiles[i], 4*2048, 4, "data/datasets/dataset_02/",
#                     128, start_clip=1145, save_index=totalfiles)

gen = DatasetGenerator("data/datasets/dataset_01/", shuffle=True)
X = gen.__getitem__(0)
plt.figure(figsize=(60, 60))
print(X[0].shape)
plt.imshow(X[0])
plt.show()