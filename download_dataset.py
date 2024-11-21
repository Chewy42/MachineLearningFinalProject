import os
import subprocess
import zipfile

dataset_dir = "dataset"

if os.path.isdir(dataset_dir):
    print("dataset folder already exists")
else:
    os.mkdir(dataset_dir)
    url = "https://universe.roboflow.com/ds/upRikv1Fpl?key=2Brd22e54E"
    zip_path = os.path.join(dataset_dir, "roboflow.zip")
    
    subprocess.run(["curl", "-L", url, "-o", zip_path], check=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    os.remove(zip_path)
