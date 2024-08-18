# FedAvg
Running FedAverage algorithm on a single server for federation learning

## Pre-requisites

- `pip install -r requirements.txt`
- Please add your own datasets in the repo.
- Change the absolute path of datasets in `client.py`

## Server
- You should run `app.py` first for running the flask server

## Client

- `python3 client.py`


## Image Processing - To prepare datasets for YOLOv8 training

Files:


- 📄 `image_processor.py`: is for generating bounding boxes for existing orchid images under `/orchid_image`, the output images will under `/output` folder

- 📄 `csv2YOLO.py`:  Covert existing orchid feature csv into a single annoation file under `/annotations` folder




```
cd image_processing/
```