import cv2
import pixelqueer as queer
import argparse
import sys # More like cis am I right? (I'm not right.)
import pickle
import numpy as np

# Mirrors see a face from the camera, and project its dream of the face.
class Mirror:
    def __init__(self, cis_eigenfaces, trans_eigenfaces, trans_matrix):
        self.cis_eigenfaces = cis_eigenfaces
        self.trans_eigenfaces = trans_eigenfaces
        self.trans_matrix = trans_matrix

    # Projects a source image onto a new, gender-bent target image.
    def project(self, source_image):
        return queer.swap_gender(self.cis_eigenfaces, self.trans_eigenfaces, self.trans_matrix, source_image)

# The frame holds two mirrors, letting the user toggle between them.
class Frame:
    def __init__(self, learn_file):
        # Pull the matrices from the brain file.
        print "Unpickling brain in formaldehyde.."
        male_eigenfaces = pickle.load(learn_file)
        female_eigenfaces = pickle.load(learn_file)
        mtf_matrix = pickle.load(learn_file)
        ftm_matrix = pickle.load(learn_file)

        # Spin up mirrors.
        print "Spooling up the mirrors.."
        self.mtf_mirror = Mirror(male_eigenfaces, female_eigenfaces, mtf_matrix)
        self.ftm_mirror = Mirror(female_eigenfaces, male_eigenfaces, ftm_matrix)

        # Default is MTF ('cause I made this :D)
        self.current = self.mtf_mirror

        # Our camera is started here.
        print "Opening shutter.."
        self.cam = cv2.VideoCapture(0)

        # Temp: store dimensions here. We should really pickle 'em.
        self.eigen_w = 768
        self.eigen_h = 512

    def capture_image(self):
        # Get an image frame from the camera.
        status, source_image = self.cam.read()

        # First, we need to crop and resize the image, so that it matches the 512x768 dimension of our eigenfaces.
        # Let the original camera w and h be Cw and Ch, respectively. We want a rescaled Rw and Rh-image,
        # with the same aspect ratio as the eigenfaces (Ew and Eh.)
        # Assuming landscape mode, if Rh = Ch then Rw = (Ew/Eh) * Rh

        cam_h = source_image.shape[0]
        cam_w = source_image.shape[1]
        cropped_h = cam_h
        cropped_w = int((self.eigen_w/float(self.eigen_h)) * cropped_h) # Round down, naturally.

        face = source_image[0:cropped_h, 0:cropped_w].astype(np.float32)

        # Now to resize, so that we have dimensions Eh and Ew.
        face = cv2.resize(face, (self.eigen_h, self.eigen_w), interpolation = cv2.INTER_LINEAR)

        # Now go to grayscale and adjust so range is [-1, 1].
        face = face.mean(axis=2)
        face = (face - 127.0) / 127.0

        # Subtract the mean pixel value from the face.
        face = face - face.mean()

        return face

    def loop(self):
        while True:
            face = self.capture_image()

            altered_image = self.current.project(face)
            cv2.imshow("Video", altered_image)

            # This is necessary or the image won't display!
            keyPressed = cv2.waitKey(1)

            # Switch from MTF to FTM, as necessary.
            if keyPressed == ord('f'):
                self.current = self.mtf_mirror
            if keyPressed == ord('m'):
                self.current = self.ftm_mirror
    
parser = argparse.ArgumentParser(description="Pixelqueer mirror")
parser.add_argument("brain", help="Path to brain file (created by cmd.py)")
args = parser.parse_args(sys.argv[1:])

# Create the frame and run forever within it.
frame = Frame(open(args.brain, "rb"))
frame.loop()
