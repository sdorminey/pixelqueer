import cv2
import pixelqueer as queer

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
        male_eigenfaces = pickle.load(learn_file)
        female_eigenfaces = pickle.load(learn_file)
        mtf_matrix = pickle.load(learn_file)
        ftm_matrix = pickle.load(learn_file)

        # Spin up mirrors.
        self.mtf_mirror = Mirror(male_eigenfaces, female_eigenfaces, mtf_matrix)
        self.ftm_mirror = Mirror(female_eigenfaces, male_eigenfaces, ftm_matrix)

        # Default is MTF ('cause I made this :D)
        self.current = self.mtf_mirror

        # Our camera is started here.
        self.cam = cv2.VideoCapture(0)

    def loop(self):
        while True:
            status, source_image = self.cam.read()
            altered_image = self.current.project(source_image)
            cv2.imshow("Video", altered_image)

            # This is necessary or the image won't display!
            cv2.waitKey(1)
    
parser = argparse.ArgumentParser(description="Pixelqueer mirror")
parser.add_argument("brain", help="Path to brain file (created by cmd.py)")
args = parser.parse_args(sys.argv[1:])

# Create the frame and run forever within it.
frame = Frame(args.brain)
frame.loop()
