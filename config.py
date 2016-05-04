import json

class Config:
    def __init__(self, path = "config.json"):
        config = None
        with open(path, "rb") as f:
            config = json.loads(f.read())

        self.face_classifier_file = config["FaceClassifierFile"]
        self.brain_file = config["BrainFile"]
        self.eigen_w = config["EigenW"]
        self.eigen_h = config["EigenH"]
        self.face_min_w = config["FaceMinW"]
        self.face_min_h = config["FaceMinH"]

