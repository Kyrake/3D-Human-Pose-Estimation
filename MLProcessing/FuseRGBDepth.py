import os
import json
import numpy as np
import opencv

keypointlist = []
directory_rgb = "/home/kyra/openpose/rgb"
directory_depth = "/home/kyra/openpose/depth"
i = 1
for filename in sorted(os.listdir(directory_rgb), key = lambda x: int(x.split("_")[0])):
    with open(os.path.join(directory_rgb,filename), "r") as infile:
        keypointRGB = json.load(infile)

for filename in sorted(os.listdir(directory_depth), key=lambda x: int(x.split("_")[0])):
    with open(os.path.join(directory_depth, filename), "r") as infile:
        keypointDepth = json.load(infile)


for i in range(len(keypointRGB)):
    for j in range(len(keypointDepth)):
        if keypointRGB(i) == keypointDepth(j):
            keypointlist.append((keypointRGB(i), keypointDepth(j)))

keypointArray = np.asarray(keypointlist)
np.save('rbgdKeypoints.npy',keypointArray)