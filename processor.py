import cv2
import numpy as np

class FaceCoord:
    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]
        self.w = coord[2]
        self.h = coord[3]

# Extracts and inserts face feature vectors into and out of images.
class FaceProcessor:
    def __init__(self, config):
        self.config = config
        self.face_classifier = cv2.CascadeClassifier(self.config.face_classifier_file)

    def locate_face(self, image):
        # Find the face in the image using the cascade classifier.
        all_face_coords = self.face_classifier.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        if len(all_face_coords) < 1:
            return None

        # Pick the largest candidate face by area.
        face_coords = FaceCoord(max(all_face_coords, key = lambda c: c[2] * c[3]))
        
        return face_coords

    def extract_region(self, image, region):
        return image[region.y:(region.y + region.h), region.x:(region.x + region.w)]

    def apply_elliptical_mask(self, image):
        w, h = image.shape
        centroid = (w/2, h/2)
        axes = (w/2, h/2)

        # Mask the source image with the elliptical area as 1.
        mask = np.zeros(image.shape, image.dtype) 
        cv2.ellipse(mask, centroid, axes, 0, 0, 360, 255, -1)

        image = cv2.bitwise_and(image, image, mask = mask)

        return image

    def to_feature(self, image):
        # Resize image to feature vector size.
        image = cv2.resize(image, (self.config.eigen_w, self.config.eigen_h), interpolation = cv2.INTER_LINEAR)

        # Rescale to [-1, 1] and subtract the mean pixel value from the image.
        image = (image - 127.0) / 127.0
        image = image - image.mean()
        return image.flatten()

    def from_feature(self, image, new_size):
        unflattened = image.reshape((self.config.eigen_h, self.config.eigen_w))
        unflattened += 1
        unflattened *= 128
        unflattened -= 1
        unflattened = unflattened.astype(np.uint8)
        resized = cv2.resize(unflattened, new_size, interpolation = cv2.INTER_LINEAR)
        print new_size
        print resized.shape
        return resized

    # Inserts a face back into the image at the specified coordinates.
    def insert_face(self, image, face, coords):
        assert face.shape[0] == coords.w and face.shape[1] == coords.h

        w, h = face.shape
        # Create a mask the size of the image where all but the ellipse is 1.
        mask = np.ones(image.shape, image.dtype) * 255
        cv2.ellipse(mask, (coords.x + w/2, coords.y + h/2), (w/2, h/2), 0, 0, 360, 0, -1)
        # Create elliptical hole in image.
        image = cv2.bitwise_and(image, image, mask = mask)

        # Add the face back in, where the hole is.
        image[coords.y:(coords.y + coords.h), coords.x:(coords.x + coords.w)] += face
        return image

