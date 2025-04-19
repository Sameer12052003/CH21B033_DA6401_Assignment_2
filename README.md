# CH21B033_DA6401_Assignment_2

## Name : Sameer Deshpande

## Roll No : CH21B033

This repository contains the implemenatation of part A and part B of assignment 2 of the course 'Introduction to Deep Learning' DA6401.

It tunes a custom CNN, trains, validates and tests it on the inaturalist dataset. It also finetunes GoogLeNet model on the same dataset. It compares performance of these two models.

## Folder Strucuture
<pre> 
CH21B033_DA6401_Assignment_2/ 

├── Part_A/ 
    ├── best_model_metrics_and_path/ 
        ├── Custom_best_model.pth               # Stores best custom model's weights
        ├── Metrics_Custom_Model.csv            # Stores best custom model's performance
    ├── custom_model/
        ├── model.py                            # Custom model architecture code 
    ├── data_preprocessing/
        ├── custom_dataset.py                   # Pytorch dataset creation script 
        ├── data_split.py                       # Splits training data into validation data
    ├── inaturalist_12K/                        # Dataset 
        ├── test 
        ├── train
        ├── val
    ├── wandb_sweeps_and_best_model/ 
        ├── best_model.py                       # Best model training and validation
        ├── sweep.py                            # Sweep configs and best model tracking
    ├── grid_plotting.py                        # Script for plotting $10 \times 3$ grid 


├── Part_B/ 
    ├── best_model_metrics_and_path/ 
        ├── GoogLeNet_best_model.pth            # Stores GoogLeNet's weights
        ├── Metrics_GoogLeNet.csv               # Stores GoogLeNet's performance    
    ├── data_preprocessing/ 
        ├── custom_dataset.py                   # Pytorch dataset creation script 
        ├── data_split.py                       # Splits training data into validation data
    ├── inaturalist_12K/                        # Dataset 
        ├── test 
        ├── train
        ├── val    
    ├── finetune.py                             # Script for fine-tuning GoogLeNet model  
    
├── README.md                                   # Project overview and instructions 
├── requirements.txt                            # Python dependencies </pre> 


## Installation Instructions

1\) Clone the repository:
```bash
git clone https://github.com/Sameer12052003/CH21B033_DA6401_Assignment_2.git
cd CH21B033_DA6401_Assignment_2
```

2\) Please create a python virtual environment: 
```bash
python -m venv venv
```

3\) Activate the python environment:
```bash
source venv/Scripts/activate
```

4\) Install all the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

Download the inaturalist dataset from: https://storage.googleapis.com/wandb_datasets/nature_12K.zip

Unzip the dataset into a folder named 'inaturalist_12K' in folders Part_A and Part_B both.

# Part A
To create validation data out of training data

```bash
cd Part_A
cd data_preprocessing
python data_split.py
```

# PartB
To create validation data out of training data
```bash

cd Part_B
cd data_preprocessing
python data_split.py
```

### Usage Instructions

# Part A

```bash
# To run the sweep code for hyperparameter tuning
cd Part_A
cd wandb_sweeps_and_best_model
export PYTHONPATH
python sweep.py

# To train, validate and test the best custom model
cd Part_A
cd wandb_sweeps_and_best_model
export PYTHONPATH
python best_model.py

# To plot the 10 * 3 grid
cd Part_A
export PYTHONPATH
python grid_plotting.py
```

# Part B
```bash
# To finetune, evaluate and test the GoogLeNet model 
cd Part_B
python finetune.py
```

## Results

Custom CNN model

| Metric            | Accuracy (%)         |
|-------------------|----------------------|
| Train Accuracy     | 28.21               |
| Validation Accuracy| 30.32               |
| Test Accuracy      | 30.40               |

Pre-trained GoogLeNet model

| Metric            | Accuracy (%)         |
|-------------------|----------------------|
| Train Accuracy     | 69.4               |
| Validation Accuracy| 68.43               |
| Test Accuracy      | 69.15              |


## Notes

Part A and Part B of this assignment contain only one single README.md file that is present in the root directory of this project.