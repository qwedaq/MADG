# MADG

This repository contains the code for the work "MADG: Margin-based Adversarial Learning for Domain Generalization."

# Dependencies: 
Please look into the requirements.txt file for the dependecies.

# Dataset & preprocessing: 
The following is the link to download the OfficeHome dataset - https://www.hemanthdv.org/officeHomeDataset.html; data_loader.py file is used to preprocess data and generate data loaders. 

# Model: 
The MDD.py file consists of the MADG model.

# Run: 
Use the train.py file to train the MADG model. Please enter the root path for the dataset in the train.py file. Before, training create folder to save the log files. To start training, enter the log folder path in train.sh file and please run the command "bash train.sh" which will log the details in the log folders depending on the target domain. After the training is finished, it evaluates the target accuracy and prints it.

# config: 
The optimizer and other training details are specified in the dann.yml file.

