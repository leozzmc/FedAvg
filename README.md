# FedAvg
Running FedAverage algorithm on a single server for federation learning

## Pre-requisites

- `pip install -r requirements.txt`
- Please add your own datasets in the repo.
- Change the absolute path of datasets in `client.py`

## Dataset Prepration

All datasets are under `clients` folder, 
- `client1` is a representative of big company
- `client2` is a representative of iof small company


### Photo Labeling

I simply use **Robeflow** to label the orchid images

> Roboflow Link: https://universe.roboflow.com/kevinliu/orchid_label

### 



## Server
- You should run `app.py` first for running the flask server

## Client

- `python3 client.py`


## Sample Traning Results

![PR_curve](https://hackmd.io/_uploads/Sy4-6ZSp0.png)

![R_curve](https://hackmd.io/_uploads/ByEZ6brTR.png)

![results](https://hackmd.io/_uploads/r1VZpZS60.png)

![val_batch0_labels](https://hackmd.io/_uploads/r1ymp-HpA.jpg)


# Program Flow

```mermaid
sequenceDiagram
    participant Client1
    participant Client2
    participant Server

    Note over Server: Listen to incoming requests
    Client1->>Client1: Train local model
    Client2->>Client2: Train local model

    Client1->>Server: Upload local model weight
    Client2->>Server: Upload local model weight

    Server->>Server: Aggregate model weights

    Server->>Client1: Notify clients of global weight availability
    Server->>Client2: Notify clients of global weight availability

    Client1->>Server: Download global weight
    Client2->>Server: Download global weight

    Note over Client1, Client2: Repeat for multiple iterations

```