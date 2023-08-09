# Traffic-Sign-Classification

## Setup For Python:
1.	Install Python
   
## Training the Model:
1.	Download the data from kaggle (https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
2.	Clone this Traffic Sign Classification repository
3.	Install all the required libraries using the command - pip install -r .\requirements.txt
4.	Run the traffic_signal_recognition.py file
5.	The model will be saved as TSR.h5 file.
6.	Run the classes.py file
7.	Test the model by giving it image of any traffic sign (images from Test data).
Respective Description of the Traffic Sign will be returned by the model.

## About -
"Traffic signs provide valuable information to drivers and other road users. They represent rules that are in 
place to keep you safe, and help to communicate messages to drivers and pedestrians that can maintain order 
and reduce accidents. Neglecting them can be dangerous.
Most signs make use of pictures, rather than words, so that they are easy to understand and can be 
interpreted by people who speak a variety of languages. For this reason, it’s important that you know what 
each picture represents, and that you use them to inform your driving. Failing to do so could result in a serious 
accident or a fine"

## Problem Statement – 
We have to develop a model which would help people learn about one of most 
underrated, yet very important part of our daily life, a traffic sign. This model has been made using deep 
learning libraries TensorFlow and its high-level API, Keras. The objective of this model is to attain an 
accuracy so strong that an individual should be able to use this model without any hesitation.

## Purpose – 
In the past and recent times, there have been many road accidents where the main reason 
for these being inadequate knowledge of road and traffic signs, it was found out that the second most 
heard reason was an individual not knowing what a particular traffic sign means. Our model focuses on 
detecting traffic signs and giving description about it, when provided an image to it through deep 
learning.

## Evaluation -
Our Goal is to create a Machine learning model which will take images and gives prediction with confidence.
It needs major understanding of CNN(Convolutional Neural Network), Python. 
So, Evaluation of Traffic Sign Classification starts from the study of Machine Learning.
First step of this project is to download a dataset (German traffic Sign Recognition Benchmark) from Kaggle. We 
have 43 different types of classes. Train the model using these images of 43 classes using CNN. Save the model as 
.h5 file

## Limitations -
Although, there are many advantages of traffic sign classification, there are certain difficulties as well. It 
may happen that the traffic sign is hidden behind the trees or any board at the road side which may cause 
the inaccurate detection and classification of traffic sign. Sometimes it may happen that the vehicle went 
so fast, that it did not detect the traffic sign. This may be dangerous and can lead to accidents. There is a 
need for further research to deal with these issues.

## Future Scope -
Traffic Signs are useful to all the individuals who are driving a vehicle on the road. Traffic Signs guide the 
drivers for following all the traffic rules and avoid any disruption to the pedestrians. The environmental 
constraints including lighting, shadow , distance (sign is quite far), air pollution, weather conditions in 
addition to motion blur, and vehicle vibration which are common in any real time system may affect the 
detection and thus the classification. Hence, there is a need for further research and advancements to 
deal with these issues. Also, there are certain traffic signs that may not be predicted accurately. For this, 
augmentation and one hot encoding techniques can be used. Augmentation involves shifting of the image, 
zoom in and rotate the images (if required).
This system helps the driver to observe the sign close to his / her eyes on the screen. This saves the time 
and efforts in manually checking whether any traffic sign board is there, identifying what type of sign it is 
and act accordingly. Traffic Sign Classification, thus, has a wide application in building smarter cars like 
automatic driving cars, where the system automatically detects, recognizes a traffic sign, and displays it.

## Conclusion -
The proposed system is simple and does the classification quite accurately on the GTSRB dataset and 
finally the model can successfully capture images and predict them accurately even if the background of 
the image is not much clear. The proposed system uses Convolutional Neural Network (CNN) to train the 
model. The final accuracy on the test dataset is 94% .The benefits of “Traffic Sign classification and 
detection system” are generally focused on driver convenience. Despite the advantages of traffic sign 
classification, there are drawbacks. There can be times when the traffic signs are covered or not visible 
clearly. This can be dangerous as the driver will not be able to keep a check on his vehicle speed and can 
lead to accidents, endangering other motorists or pedestrians, demanding further research
