# Car Damage Assessment using Densenet

![Car Damage Assesment](https://github.com/koushik395/Car-damage-assessment-using-Densenet/blob/main/images/output.jpeg)

## Overview:
In Car Insurance industry, a lot of money is being wasted on Claims leakage. Claims leakage is the gap between the optimal and actual settlement of a claim. Visual inspection and validation are being used to reduce claims leakage. But doing inspection might take a long time and result in delaying of claims processing. An automated system for doing inspection and validation will be of great help in speeding up the process.

## 1. Business Use Case:
To reduce Claims leakage during Insurance processing. Visual inspection and validation are being done. As this takes a long time because the person needs to come and inspect the damage. We are trying to automate this procedure. Using this automation will result in Claims processing faster.

## 2. Mapping the Problem to Deep Learning Model:
We are trying to automate the Visual inspection and validation of vehicle damage. The input data we have are car damaged images.

For Validation of Vehicle damage we will divide the problem into three stages. 
* First we check whether the given input image of car has been damaged or not.

* Second we check on which side (Front, Rear, Side) the Car in image has been damaged.

* Third we check for the Severity of damage (Minor, Moderate, Severe).

This problem is a classic classification problem and Since we will be dealing with images as input, we will be using Convolutional Neural Networks (CNN).

## 3. Data Source

## 4. Procedure
To make the insurance claims for car damage faster we have used CNN to classify the car damage and detect type of damage using yolo v3.
* Downloaded the whole dataset from Kaggle website.
* Exploratory Data Analysis of Data using Matplotlib and Seaborn.
* Since the dataset on car damage are rare I have used two types of Data Augmentation to synthetically enlarge the dataset.
* Created three types of data folders Orginal Data, Original Data + Augmentation 1, Original Data + Augmentation 2.
* Used Pretrained model Densenet without FC layers and added my own configuration of FC layers (1 Flatten layer,2 Full Connection layers and 1 Dropout layer).
* Trained two types of model for each pretrained model.
  * Training only FC layers
  * Training All layers.
* Used Logistic Regression as baseline model. Trained around 108 models.
* Trained Yolov3 to detect type of damage in Car Dent and Scratch, Smash, Glass and Light Broken.
* Densenet trained on all layers using Original Data + Augmentation 1 gave best performance over other models.
* Used Flask and Streamlit to make a web app on local machine.


https://github.com/Charishma-Bailapudi/Test/assets/98004429/207686ab-f889-46db-9a1e-be7ab1af1606


  
