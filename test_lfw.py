import os
from PIL import Image
from glob import glob
from face_group import FacesCluster
import pickle


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

faces_cluster.plot()
