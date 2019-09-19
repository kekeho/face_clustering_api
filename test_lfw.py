import os
from PIL import Image
from glob import glob
from face_group import FacesCluster


images_dir = os.path.join(os.path.dirname(__file__), 'images/lfw/*')
filelist = glob(os.path.join(images_dir, '*[.jpg | .jpeg | .JPEG | .JPG | .png | .PNG]'))

images = [Image.open(i) for i in filelist]
faces_cluster = FacesCluster(images)

faces_cluster.plot()
