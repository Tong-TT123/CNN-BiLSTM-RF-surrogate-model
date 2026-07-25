# CNN-BiLSTM-RF Surrogate Model for Capsizing Moment Prediction of Coupled Shiplift Systems
Source code for CFD-based surrogate modeling of capsizing moment prediction using CNN-BiLSTM-RF 
This repository provides the MATLAB implementation and example dataset for the proposed CNN-BiLSTM-RF surrogate model for capsizing moment prediction of coupled shiplift systems.
The proposed model combines convolutional neural network (CNN), bidirectional long short-term memory network (BiLSTM), and random forest (RF) to establish an efficient data-driven surrogate model for nonlinear dynamic response prediction.
# Overview
The proposed CNN-BiLSTM-RF framework consists of:
1. Data preprocessing and abnormal sample detection;
2. CNN-based feature extraction;
3. BiLSTM-based temporal feature learning;
4. Random forest regression;
5. Adaptive fusion of CNN-BiLSTM and RF predictions.
The model was developed using CFD-generated datasets from coupled shiplift systems.
The complete dataset and MATLAB workspace files are provided through the associated Zenodo repository.
Zenodo DOI:(Available after dataset publication)
# Repository Structure
CNN-BiLSTM-RF.m
Main program of the proposed model
CNN_feature_extract.m
CNN-based feature extraction module optimize_fitrCNN_BILSTM_att.m Optimization and training procedure for CNN-BiLSTM model
optimize_fitrtreebag1.m
Optimization and training procedure for random forest model
abnorm_detect.m
Abnormal sample detection procedure
example.xlsx
Example dataset showing input/output data format
README.md
LICENSE
# Software Requirements
The source codes were developed and tested using:
MATLAB R2023a
Required MATLAB toolboxes:
Deep Learning Toolbox
Statistics and Machine Learning Toolbox
# Dataset Description
The complete dataset was generated using CFD simulations of coupled shiplift systems.
The dataset contains:
19 independent shiplift models;
19,000 samples in total.
The example file:
example.xlsx
only provides the organization format of the input and output variables.
The complete CFD dataset is available through the Zenodo repository.
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
The prediction target is:
M(t+1)
which represents the capsizing moment at the next sampling instant.
# Data Processing
Before model training, the following preprocessing procedures are performed:
1. Abnormal sample detection;
2. Data normalization using Z-score normalization;
3. Dataset division into training, validation, and testing sets.
The dataset is divided as:
Dataset Samples Ratio 
Training 15000 80% 
Validation 1875 10% 
Testing 1875 10% 
# Model Configuration
## CNN Feature Extraction
The CNN module is used to extract nonlinear feature representations from the original input variables.
The extracted CNN features are combined with the original variables to enhance model representation capability.
## BiLSTM Network
The BiLSTM network is used to capture temporal dependencies from the extracted feature sequences.
Main parameters:
Hidden units: 82
Maximum training epochs: 70
Batch size: 1875
## Random Forest
The RF model is used as an independent regression branch.
Configuration:
Number of decision trees: 200
Minimum leaf size: 1
# Optimization Strategy
The model parameters are optimized using metaheuristic optimization algorithms.
The optimization process includes:
CNN-BiLSTM parameter optimization;
Random forest parameter optimization.
The optimization configuration is stored in the MATLAB workspace file.
# MATLAB Workspace File
The MATLAB workspace file:
CNN-BiLSTM-RF.mat
contains:
processed datasets;
normalized training, validation, and testing data;
feature information;
CNN feature extraction model;
model configuration parameters;
optimization parameters;
trained model information.
Due to the large file size, the MATLAB workspace file is not uploaded to GitHub.
It is provided through the Zenodo repository.
# Running Instructions
## Step 1
Install MATLAB R2023a with required toolboxes.
## Step 2
Download this repository.
## Step 3
Prepare the dataset according to the format shown in:
example.xlsx
## Step 4
Run:
CNN-BiLSTM-RF.m
The program automatically performs:
data preprocessing;
abnormal sample detection;
CNN feature extraction;
CNN-BiLSTM model training;
RF model training;
prediction fusion;
performance evaluation.
# Reproducibility
To reproduce the reported results:
1. Download the GitHub repository;
2. Download the complete dataset and MATLAB workspace files from Zenodo;
3. Install MATLAB R2023a;
4. Run:
CNN-BiLSTM-RF.m
All source codes, datasets, and configuration files required for reproduction are provided.
# Citation
If you use this code or dataset, please cite:
(Article citation information will be added after publication)
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
