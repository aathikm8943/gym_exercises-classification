import os 
import sys
from pathlib import Path

list_of_files = [
    "main.py",
    "config.py",
    "src/components/preprocessing.py",
    "src/components/dataloader.py",
    "src/components/model.py",
    "src/components/evaluate.py",
    "src/components/__init__.py",
    "src/helpers/__init__.py",
    "src/helpers/extract_zip.py",
    "src/loggingInfo/loggingFile.py",
    "src/loggingInfo/__init__.py",
    "src/app/__init__.py",
    "src/app/streamlit_app.py",
    "data/input_video",
    ".gitignore",
    "experiments/experiment.ipynb",
    "requirements.txt",
    "Readme.md"
]

for file in list_of_files:
    fileFullPath = Path(file)
    fileExt, fileName = os.path.split(fileFullPath)
    
    if fileExt != "":
        os.makedirs(fileExt, exist_ok=True)
    
    if not (os.path.exists(fileFullPath)):
        
        with open(fileFullPath, "wb") as f:
            pass
    