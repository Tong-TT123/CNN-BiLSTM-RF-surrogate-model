# CNN-BiLSTM-RF Surrogate Model for Capsizing Moment Prediction of Coupled Shiplift Systems
Source code for CFD-based surrogate modeling of capsizing moment prediction using CNN-BiLSTM-RF 
This repository provides the MATLAB implementation and example dataset for the proposed CNN-BiLSTM-RF surrogate model for capsizing moment prediction of coupled shiplift systems.
The complete dataset and MATLAB workspace files are available through the Zenodo repository.
Zenodo DOI:
(Available after dataset publication)
# Repository Contents
The repository contains:
CNN-BiLSTM-RF.m
Main program for running the proposed surrogate model.
CNN_feature_extract.m
CNN-based feature extraction function.
optimize_fitrCNN_BILSTM_att.m
Optimization function for CNN-BiLSTM model.
optimize_fitrtreebag1.m
Optimization function for random forest model.
abnorm_detect.m
Abnormal sample detection function.
example.xlsx
Example dataset showing the input and output data format.
README.md
Description of the repository.
LICENSE
License information.
# Software Requirements
The source codes were developed and tested using:
MATLAB R2023a
# Dataset Description
The dataset was generated from CFD simulations of coupled shiplift systems.
The complete dataset contains:
19 shiplift models;
19,000 samples.
The dataset consists of input variables and corresponding capsizing moment responses.
The file `example.xlsx` provides the data organization format and variable arrangement.
The complete dataset and MATLAB workspace files are provided through the Zenodo repository.
# Input and Output Variables
The model input contains 10 variables:
1. L2/L1
2. B2/B1
3. H2/H
4. H/H1
5. M(t-1)
6. M(t)
7. F(t-1)
8. F(t)
9. a(t-1)
10. a(t)
The prediction target is:M(t+1) representing the capsizing moment at the next sampling instant.
# Data Processing
The data processing procedure includes:
abnormal sample detection;
data normalization;
dataset division for model training and evaluation.
The dataset is divided as:
Dataset Samples Ratio 
Training 15000 80% 
Validation 1875 10% 
Testing 1875 10% 
# Running Instructions
Install MATLAB R2023a.
Download this repository.
Prepare the dataset according to the format shown in:example.xlsx
Run:CNN-BiLSTM-RF.m
# Reproducibility
1. Download the source code from this repository.
2. Download the complete dataset and MATLAB workspace files from Zenodo.
3. Run the main MATLAB script:CNN-BiLSTM-RF.m
# Citation
Dataset DOI:
(Available after Zenodo publication)
# Contact
Yang Zhang
Institute of Engineering and Technology, Hubei University of Science and Technology
Xianning 437100, P.R. China
Email:
zhangyang619921@qq.com
# License
This project is released under the license provided in the LICENSE file.
