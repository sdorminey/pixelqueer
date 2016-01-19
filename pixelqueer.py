from os import path, listdir
import sys

import scipy
from scipy import misc
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import Image as pil

def loadface(image_path):
    face = misc.imread(image_path).astype(np.float32)
    face = face.mean(axis=2)
    # Subtract the mean pixel value from the face.
    face = face - face.mean()
    return face

def eigenfaces(image_paths):
    face_matrix = np.array([loadface(f).flatten() for f in image_paths])
    print "Face matrix:", face_matrix.shape, face_matrix.size

    # According to wikipedia on eigenfaces, the upper singular values are equivalent
    # to the eigenvectors of the autocorrelation matrix of images!
    U, V, T = lin.svd(face_matrix.transpose(), full_matrices=False)
    
    return U

def project(basis, vector):
    return basis * vector

def swap_gender(training_data_path, source_image):
    male_path = path.abspath(path.join(training_data_path, "Male"))
    female_path = path.abspath(path.join(training_data_path, "Female"))
    male_faces = [path.abspath(path.join(male_path, f)) for f in listdir(male_path)]
    female_faces = [path.abspath(path.join(female_path, f)) for f in listdir(female_path)]

    # Build the eigenfaces for the male and female images in the training set.
    male_eigenfaces = eigenfaces(male_faces)
    print "Male eigenfaces:", male_eigenfaces.shape, male_eigenfaces.size
    female_eigenfaces = eigenfaces(female_faces)

    # Load the source image and project it onto the male eigenfaces.
    source_face = loadface(source_image)
    source_male_parameters = male_eigenfaces.transpose() * source_face.flatten()

    # Visualize how the projection looks!
    target_image = male_eigenfaces.transpose() * source_male_parameters
    target_image = target_image.sum(axis=0)
    print "Target shape:", target_image.shape, "Source shape:", source_face.shape
    target_image = target_image.reshape(source_face.shape)
    plt.imshow(target_image)
    plt.show()

print "Hello world!"
swap_gender(sys.argv[1], sys.argv[2])
