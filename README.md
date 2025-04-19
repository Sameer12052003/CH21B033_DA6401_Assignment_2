# CH21B033_DA6401_Assignment_2

## Folder Strucuture

<pre> 
project-root/ 

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