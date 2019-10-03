import os
from PIL import Image
from glob import glob
from face_group import FacesCluster
import pickle
from collections import Counter


images_dir = os.path.join(os.path.dirname(__file__), 'images/lfw/*')
filelist = glob(os.path.join(images_dir, '*[.jpg | .jpeg | .JPEG | .JPG | .png | .PNG]'))

pickle_file = 'test_lfw_cluster.pkl'
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        faces_cluster = pickle.load(f)
    
    # Re clustering
    faces_cluster._clustering(0.55, 1, 'compressed_id')
else:
    images = [Image.open(i) for i in filelist]
    faces_cluster = FacesCluster(images, 'compressed_id', 1, 2)
    with open(pickle_file, 'wb') as f:
        pickle.dump(faces_cluster, f)


# faces_cluster.plot()
print('CLUSTER LEN', faces_cluster.cluster_len)
print('JUNK COUNT', faces_cluster.junk_count)
print('JUNK PER', faces_cluster.junk_per)
print('NOISE PER AVERAGE', faces_cluster.noise_per_average)

# Duplicate cluster count
groups = [g.suggested_name for g in faces_cluster.group if g.status == 'GOOD']
duplicate_count = len([None for k, v in Counter(groups).items() if v > 1])
print('DUPLICATE', duplicate_count)
