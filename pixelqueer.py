from os import path, listdir
import sys
import argparse
import pickle

import scipy
from scipy import misc
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL as pil
from PIL import Image

def loadimage(image_path):
    face = misc.imread(image_path).astype(np.float32)
    # Adjust so range is [-1, 1].
    face = face.mean(axis=2)
    face = (face - 127.0) / 127.0
    # Subtract the mean pixel value from the face.
    face = face - face.mean()
    return face

def eigenfaces(face_matrix, max_faces):
    print "Face matrix:", face_matrix.shape, face_matrix.size

    # According to wikipedia on eigenfaces, the upper singular values are equivalent
    # to the eigenvectors of the autocorrelation matrix of images!
    U, V, T = lin.svd(face_matrix.transpose(), full_matrices=False)
    
    # U     e x w   Eigenface vectors as rows.
    eigenfaces = np.array(U.transpose()[0:max_faces,:])
    print "Eigenfaces:", eigenfaces.shape
    
    return eigenfaces

def learn_faces(male_faces, female_faces):
    male_eigenfaces = eigenfaces(male_faces)
    female_eigenfaces = eigenfaces(female_faces)

    return (male_eigenfaces, female_eigenfaces)

def learn_transition(trans_eigenfaces, cis_eigenfaces, training_images):
    # trans_eigenfaces      e_t x w     Eigenface basis for the opposite gender.
    # cis_eigenfaces        e_c x w     Eigenface basis for the original gender.
    # training_images       w   x s     Training images (as rows)
    print "Training images:", training_images.shape

    cis_correct_values = np.dot(cis_eigenfaces, training_images.transpose()).transpose()
    print "Correct Values:", cis_correct_values.shape

    trans_starting_values = np.dot(trans_eigenfaces, training_images.transpose()).transpose()
    print "Starting values:", trans_starting_values.shape

    transition_matrix, residuals, rank, singular_values = lin.lstsq(trans_starting_values, cis_correct_values)
    transition_matrix = transition_matrix.transpose()
    print "Residuals:", residuals
    print "Transition matrix:", transition_matrix.shape
    print "Rank:", rank

    return transition_matrix

# Loads face vectors
def load_faces(root_path, subfolder_name, max_faces):
    subfolder_path = path.abspath(path.join(root_path, subfolder_name))
    image_paths = [path.join(subfolder_path, f) for f in listdir(subfolder_path)]
    image_paths = image_paths[:max_faces]
    images = [loadimage(p) for p in image_paths]
    return (np.array([i.flatten() for i in images]), images[0].shape)

def learn(training_data_path, max_eigenfaces, max_faces):
    male_faces, image_shape = load_faces(training_data_path, "Male", max_faces)
    female_faces, _ = load_faces(training_data_path, "Female", max_faces)

    # Build the eigenfaces for the male and female images in the training set.
    male_eigenfaces = eigenfaces(male_faces, max_eigenfaces)
    female_eigenfaces = eigenfaces(female_faces, max_eigenfaces)
    print "Female faces:", female_faces.shape

    mtf_matrix = learn_transition(male_eigenfaces, female_eigenfaces, female_faces)
    ftm_matrix = learn_transition(female_eigenfaces, male_eigenfaces, male_faces)

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

def alter_image(source_eigenfaces, target_eigenfaces, transition_matrix, source_image):
    original_image = loadimage(source_image)
    altered_image = swap_gender(source_eigenfaces, target_eigenfaces, transition_matrix, source_image)
    return altered_image

def plot_image(original_image, altered_image):
    f = plt.figure()
    f.add_subplot(2, 1, 1)
    plt.imshow(original_image, cmap = cm.Greys_r)
    f.add_subplot(2, 1, 2)
    plt.imshow(altered_image, cmap = cm.Greys_r)
    plt.show()

def run(args):
    if args.action == "learn":
        male_eigenfaces, female_eigenfaces, mtf_matrix, ftm_matrix = learn(args.source, args.maxEigenfaces, args.maxImages)
        learn_file = open(args.brain, "wb")
        pickle.dump(male_eigenfaces, learn_file)
        pickle.dump(female_eigenfaces, learn_file)
        pickle.dump(mtf_matrix, learn_file)
        pickle.dump(ftm_matrix, learn_file)

    elif args.action == "run":
        learn_file = open(args.brain, "rb")
        male_eigenfaces = pickle.load(learn_file)
        female_eigenfaces = pickle.load(learn_file)
        mtf_matrix = pickle.load(learn_file)
        ftm_matrix = pickle.load(learn_file)

        original_image = args.image
        altered_image = None
        print args.direction
        if args.direction == "mtf":
            print "MTF Mode"
            altered_image = alter_image(male_eigenfaces, female_eigenfaces, mtf_matrix, original_image)
        elif args.direction == "ftm":
            print "FTM Mode"
            altered_image = alter_image(female_eigenfaces, male_eigenfaces, ftm_matrix, original_image)
        else:
            print "Non-binary gender not yet supported :("
            return

        if args.out is not None:
            altered_image_to_save = Image.fromarray(((altered_image+1.0)/2.0 * 255).astype(np.uint8))
            altered_image_to_save.save(args.out)
        else:
            plot_image(original_image, altered_image)


print "Hello world!"

parser = argparse.ArgumentParser(description="Genderbending!")
parser.add_argument("action", choices=["learn", "run"])
parser.add_argument("brain", help="Destination file for learned data")
parser.add_argument("--source", help="Source image folder containing Male and Female subfolders")
parser.add_argument("--direction", choices=["mtf", "ftm"], help="Direction to bend gender")
parser.add_argument("--maxEigenfaces", help="Maximum number of eigenfaces to use", default=max, type=int)
parser.add_argument("--maxImages", help="Maximum number of images to use (per gender)", default=max, type=int)
parser.add_argument("--image", help="Image to load (run mode only)")
parser.add_argument("--out", help="Output file path")

args = parser.parse_args(sys.argv[1:])
run(args)
