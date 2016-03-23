import xml.etree.ElementTree as ElementTree
import sys # down with sys!
import os
import shutil

def copy_file(source_file, target_folder):
    print source_file, target_folder
    shutil.copy(source_file, target_folder)

disc_mappings = {"Disc1": "dvd1", "Disc2": "dvd2"}

root_location = sys.argv[1]
target_location = sys.argv[2]

doc = ElementTree.parse(os.path.join(root_location, "dvd1", "data", "ground_truths", "xml", "subjects.xml")).getroot()
gender_by_subject = {}
for subject in doc.findall("Subject"):
    sid = subject.get("id")
    gender = subject.find("Gender").get("value")
    gender_by_subject[sid] = gender

doc = ElementTree.parse(os.path.join(root_location, "dvd1", "data", "ground_truths", "xml", "recordings.xml")).getroot()
for recording in doc.findall("Recording"):
    url = recording.find("URL")
    path = os.path.join(root_location, disc_mappings[url.get("root")], url.get("relative"))
    subject = recording.find("Subject")
    sid = subject.get("id")
    pose = subject.find("Application").find("Face").find("Pose").get("name")

    if pose != "fa":
        continue

    copy_file(path, os.path.join(target_location, gender_by_subject[sid]))
