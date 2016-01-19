import os
import sys

import scipy
from scipy import misc
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import Image as pil

path = sys.argv[1]
male_path = os.path.abspath(os.path.join(path, "Male"))
female_path = os.path.abspath(os.path.join(path, "Female"))
male_faces = [os.path.abspath(os.path.join(male_path, f)) for f in os.listdir(male_path)]
female_faces = [os.path.abspath(os.path.join(female_path, f)) for f in os.listdir(female_path)]

def eigenfaces(images):
    faces = None
    face_shape = None
    for k in xrange(len(images)):
        image = images[k]
        print image
        face = misc.imread(image).astype(np.float32)
        if face_shape is None:
            face_shape = face.shape
        face = face.mean(axis=2)
        face = face - face.mean()
        face_vector = face.flatten()
        if faces is None:
            faces = np.zeros((face_vector.size, len(images)))
        faces[:, k] = face_vector[:]
    print faces.size
    print faces.shape
    U, V, T = lin.svd(faces, full_matrices=False)
    
    for k in xrange(len(images)):
        print U.shape
        print face_shape
        eigenface = U[:, k].reshape((face_shape[0], face_shape[1]))
        plt.imshow(eigenface)
        plt.show()
    return eigenfaces

print "Hello world!"
#male_eigenfaces = eigenfaces(male_faces)
female_eigenfaces = eigenfaces(female_faces)
