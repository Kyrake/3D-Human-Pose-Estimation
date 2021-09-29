import os
import json
import numpy as np

keypointlist = []
directory = "/home/kyra/openpose/testo1"
i = 1
for filename in sorted(os.listdir(directory), key = lambda x: int(x.split("_")[0])):
    with open(os.path.join(directory,filename), "r") as infile:
        keypointJson = json.load(infile)
        peopleJson = keypointJson["people"]

        if (not len(peopleJson)):
            continue
        people0 =  peopleJson[0]
        peopleKeypoints = people0["pose_keypoints_2d"]

        keypointlist.append(peopleKeypoints)
        keypointArray =  np.asarray(keypointlist)
        #print(keypointArray.shape)


#json_str = json.dumps(keypointlist)

with open('keypoints5.txt', 'w') as f:
    f.write(str(keypointlist))
with open('keypoints5.json', 'w') as f:
    json.dump(keypointlist, f)
np.save('keypoints5.npy',keypointArray)

