import os
import sys

import scipy
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import Image as pil

path = sys.argv[1]
male_path = os.path.abspath(os.path.join(path, "Male"))
female_path = os.path.abspath(os.path.join(path, "Female"))
male_faces = [os.path.abspath(os.path.join(male_path, f)) for f in os.listdir(male_path)]
female_faces = [os.path.abspath(os.path.join(female_path, f)) for f in os.listdir(female_path)]

def eigenfaces(images):
    for image in images:
        print image
        face = misc.imread(image).astype(np.float32)
        face = face.mean(axis=2)
        face = face - face.mean()
        print face.size, face.shape
        plt.imshow(face)
        plt.show()

print "Hello world!"
#male_eigenfaces = eigenfaces(male_faces)
female_eigenfaces = eigenfaces(female_faces)
