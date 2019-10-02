# Copyright (c) 2019 Hiroki Takemura (kekeho)
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib
matplotlib.use('WebAgg')

import face_recognition
import umap
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from typing import List

def _check_same_shape(images: np.array):
    first = images[0].shape
    for i in images[1:]:
        if i.shape != first:
            return False
    
    return True


class FaceObj(object):
    def __init__(self, filename: str, location: np.array, id_array: np.array):
        self.filename = filename
        self.id_array = id_array
        self.location = location
        self.compressed_id = None
        self.group_id = None


class FacesCluster(object):
    def __init__(self, images: List[Image.Image], data_choice: str, n_neighbors: int, n_components: int):
        self.images = images
        self.np_images = list(map(np.asarray, self.images))
        self.filenames = list(map(lambda x: x.filename, self.images))
        self.faces = self._calc_face_encoding(n_neighbors, n_components)

        self._clustering(0.55, 1, data_choice)
        self.group = list(self)
        self.noise = [face for face in self.faces if face.group_id == -1]


    def __len__(self):
        return self.cluster_len
    

    def __iter__(self):
        for i in range(0, self.cluster_len):
            yield [face for face in self.faces if face.group_id == i]


    def _calc_face_encoding(self, n_neighbors: int, n_components: int):

        # Calc face encodings
        faces_list = []
        if _check_same_shape(self.np_images):
            location_list = face_recognition.api.batch_face_locations(list(self.np_images))
        else:
            location_list = [face_recognition.api.face_locations(img, model='cnn') for img in self.np_images]

        for image, locations, filename in zip(self.np_images, location_list, self.filenames):
            faces = face_recognition.face_encodings(image, locations)
            for face, location in zip(faces, locations):
                face_obj = FaceObj(filename, location, face)
                faces_list.append(face_obj)
        
        # Downsizing face encodings 128d to 3d
        fit = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
        compressed = fit.fit_transform([x.id_array for x in faces_list])
        for i, comp in enumerate(compressed):
            faces_list[i].compressed_id = comp

        return faces_list


    def _clustering(self, eps: float, min_samples: int, data_choice: str):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db_cluster = eval(f'db.fit([face.{data_choice} for face in self.faces])')
        for i, label in enumerate(db_cluster.labels_):
            self.faces[i].group_id = label
        
        self.cluster_len = np.max(db_cluster.labels_) + 1


    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        # Plot point
        for faces in self:
            points = np.array([face.compressed_id for face in faces])
            ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

            for face in faces:
                (x, y, z) = face.compressed_id
                ax.text(x, y, z, str(face.location) + ' ' + face.filename)

            # Print Group
            print('Group', faces[0].group_id)
            for face in faces:
                print(face.location, face.filename)
            print('')  # \n

        # Print noise faces
        print('Noise')
        for face in self.noise:
            print(face.location, face.filename)
        
        plt.show()
