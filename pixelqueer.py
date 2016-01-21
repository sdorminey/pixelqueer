from os import path, listdir
import sys

import scipy
from scipy import misc
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import Image as pil

def loadimage(image_path):
    face = misc.imread(image_path).astype(np.float32)
    # Adjust so range is [-1, 1].
    face = face.mean(axis=2)
    face = (face - 127.0) / 127.0
    # Subtract the mean pixel value from the face.
    face = face - face.mean()
    return face

def eigenfaces(face_matrix):
    print "Face matrix:", face_matrix.shape, face_matrix.size

    # According to wikipedia on eigenfaces, the upper singular values are equivalent
    # to the eigenvectors of the autocorrelation matrix of images!
    U, V, T = lin.svd(face_matrix.transpose(), full_matrices=False)
    
    # U     e x w   Eigenface vectors as rows.
    return U.transpose()

def learn_faces(male_faces, female_faces):
    male_eigenfaces = eigenfaces(male_faces)
    female_eigenfaces = eigenfaces(female_faces)

    return (male_eigenfaces, female_eigenfaces)

def learn_transition(trans_eigenfaces, cis_eigenfaces, training_images):
    # trans_eigenfaces      e_t x w     Eigenface basis for the opposite gender.
    # cis_eigenfaces        e_c x w     Eigenface basis for the original gender.
    # training_images       w   x s     Training images (as rows)
    print "Training images:", training_images.shape

    cis_correct_values = np.dot(cis_eigenfaces, training_images.transpose())
    print "Correct Values:", cis_correct_values.shape

    trans_starting_values = np.dot(trans_eigenfaces, training_images.transpose()).transpose()
    print "Starting values:", trans_starting_values.shape

    transition_matrix, residuals, s, singular_values = lin.lstsq(trans_starting_values, cis_correct_values)
    transition_matrix = transition_matrix.transpose()
    print "Transition matrix:", transition_matrix.shape

    return transition_matrix

# Loads face vectors
def load_faces(root_path, subfolder_name):
    subfolder_path = path.abspath(path.join(root_path, subfolder_name))
    image_paths = [path.join(subfolder_path, f) for f in listdir(subfolder_path)]
    images = [loadimage(p) for p in image_paths]
    return (np.array([i.flatten() for i in images]), images[0].shape)

def learn(training_data_path):
    male_faces, image_shape = load_faces(training_data_path, "Male")
    female_faces, _ = load_faces(training_data_path, "Female")

    # Build the eigenfaces for the male and female images in the training set.
    male_eigenfaces = eigenfaces(male_faces)
    female_eigenfaces = eigenfaces(female_faces)
    print "Female faces:", female_faces.shape

    mtf_matrix = learn_transition(male_eigenfaces, female_eigenfaces, female_faces)
    ftm_matrix = learn_transition(female_eigenfaces, male_eigenfaces, male_faces)
    ftm_matrix = None

    return (male_eigenfaces, female_eigenfaces, mtf_matrix, ftm_matrix)
    
def swap_gender(source_eigenfaces, target_eigenfaces, transition_matrix, source_image_path):
    # source_eigenfaces     e_s x w     Eigenface basis for source gender.
    # target_eigenfaces     e_t x w     Eigenface basis for target gender.
    # transition_matrix     e_s x e_t   Transitions from source basis to target basis.
    # source_face           w   x 1     Source image vector.

    # source_params         e_s x 1     Projection onto source basis.
    source_image = loadimage(source_image_path)
    source_face  = source_image.flatten()
    source_params = np.dot(source_eigenfaces, source_face)

    # target_params         e_t x 1     Transition from source params to target.
    target_params = np.dot(transition_matrix, source_params)

    # target_face           w x 1       Deprojected face.
    target_face = np.dot(target_eigenfaces.transpose(), target_params)

    # Unflatten back into 2D matrix and return.
    return target_face.reshape(source_image.shape)

print "Hello world!"
male_eigenfaces, female_eigenfaces, mtf_matrix, ftm_matrix = learn(sys.argv[1])
image = swap_gender(male_eigenfaces, female_eigenfaces, mtf_matrix, sys.argv[2])
plt.imshow(image)
plt.show()
