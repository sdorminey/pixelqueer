import cv2
import pixelqueer as queer
import argparse
import sys # More like cis am I right? (I'm not right.)
import pickle
import numpy as np
from config import *
from processor import *

# Mirrors see a face from the camera, and project its dream of the face.
class Mirror:
    def __init__(self, cis_eigenfaces, trans_eigenfaces, trans_matrix):
        self.cis_eigenfaces = cis_eigenfaces
        self.trans_eigenfaces = trans_eigenfaces
        self.trans_matrix = trans_matrix

    # Projects a source image onto a new, gender-bent target image.
    def project(self, source_image):
        #return source_image
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
        self.config = Config()
        self.p = FaceProcessor(self.config)
        self.mtf_mirror = Mirror(male_eigenfaces, female_eigenfaces, mtf_matrix)
        self.ftm_mirror = Mirror(female_eigenfaces, male_eigenfaces, ftm_matrix)

        # Default is MTF ('cause I made this :D)
        self.current = self.mtf_mirror

        # Our camera is started here.
        print "Opening shutter.."
        self.cam = cv2.VideoCapture(1)
        cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    def capture_image(self):
        # Get an image frame from the camera.
        status, frame = self.cam.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.transpose().copy()
        coords = self.p.locate_face(frame)
        if coords == None:
            if self.config.show_background:
                return frame
            return np.zeros(frame.shape, frame.dtype)

        # Debug
        #cv2.rectangle(frame, (coords.x, coords.y), (coords.x+coords.w, coords.y+coords.h), 0, 2)

        face = self.p.extract_region(frame, coords)

        face = self.p.apply_elliptical_mask(face)
        face = (face - 127.0) / 127.0
        face = face - face.mean()

        resized_face = cv2.resize(face, (self.config.eigen_w, self.config.eigen_h), interpolation = cv2.INTER_LINEAR)
        altered_face = self.current.project(resized_face)
        altered_face = cv2.resize(altered_face, face.shape, interpolation = cv2.INTER_LINEAR)
        gray_face = cv2.normalize(altered_face,altered_face,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        gray_face = self.p.apply_elliptical_mask(gray_face)
        #gray_face = ((altered_face+1.0)/2.0 * 255).astype(np.uint8)
        if self.config.show_background:
            frame = self.p.insert_face(frame, gray_face, coords)
        else:
            frame = self.p.insert_face(np.zeros(frame.shape, frame.dtype), gray_face, coords)

        return frame

    def loop(self):
        while True:
            face = self.capture_image()

            cv2.imshow("Video", face)

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
