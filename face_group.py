# Copyright (c) 2019 Hiroki Takemura (kekeho)
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib
matplotlib.use('WebAgg')

import face_recognition
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from typing import List
import json
from collections import Counter

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


class GroupObj(object):
    def __init__(self, group_id, faces: List[FaceObj]):
        self.group_id = group_id
        self.faces = faces
        self.status = None  # 'JUNK' or 'GOOD'
        self.suggested_name = None
        self.noise_per = None

        self.__check_junk_group()
    

    def __check_junk_group(self):
        namelist = [face.filename.split('/')[-2] for face in self.faces]
        counts = Counter(namelist)
        most_name, most_count = counts.most_common()[0]

        if (most_count / len(namelist)) >= 0.7:
            self.status = 'GOOD'
            self.suggested_name = most_name
            self.noise_per = 1 - (most_count / len(namelist))
        else:
            self.status = 'JUNK'


class FacesCluster(object):
    def __init__(self, images: List[Image.Image], data_choice: str, n_neighbors: int, n_components: int):
        self.images = images
        self.np_images = list(map(np.asarray, self.images))
        self.filenames = list(map(lambda x: x.filename, self.images))
        self.faces = self._calc_face_encoding(n_neighbors, n_components)

        self._clustering(0.55, 1, data_choice)
        self.group = list(self)

        self.junk_count = len([g for g in self.group if g.status == 'JUNK'])
        self.junk_per = self.junk_count / len(self.group)
        self.noise_per_average = sum([g.noise_per for g in self.group if g.status == 'GOOD']) / len(self.group)

        self.noise = [face for face in self.faces if face.group_id == -1]

    def __len__(self):
        return self.cluster_len    

    def __iter__(self):
        for i in range(0, self.cluster_len):
            faces = [face for face in self.faces if face.group_id == i]
            yield GroupObj(i, faces)

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
        fit = TSNE(n_components=n_components)
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
    

    def get_jsonstr(self):
        obj = dict()
        for faces in self:
            group_id = int(faces[0].group_id)
            face_list = [face.filename for face in faces]
            obj[group_id] = face_list

        return json.dumps(obj)


    def plot(self):
        # Plot point
        for group in self:
            points = np.array([face.compressed_id for face in group.faces])
            plt.plot(points[:, 0], points[:, 1], markersize=0.05)

            for face in group.faces:
                (x, y) = face.compressed_id
                plt.text(x, y, face.filename.split('/')[-2], fontsize=0.1)

            # Print Group
            print('Group', group.faces[0].group_id)
            for face in group.faces:
                print(face.location, face.filename)
            print('')  # \n

        # Print noise faces
        print('Noise')
        for face in self.noise:
            print(face.location, face.filename)
        
        plt.savefig('face_group.pdf')
