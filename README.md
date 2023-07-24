## Introduction
* The project goal is creating a method based on Pulsated Optically Stimulated Luminescence (POSL) and artificial intelligence (AI) algorithms to
  provide an age estimate for any sediment. We aim to provide a toolkit for rapid, highly accurate age estimation in cases
  where there are no ceramic finds in sediments.

* The program will help to quickly date, while excavating, sediments that do not have an obvious date. This is a tool with great potential to improve the fieldwork of archaeologists. 

![selution](https://github.com/shmooel28/finalProject/blob/master/utils/selution.png)


## Background information
Dating stratigraphy is a critical part of archaeological site research. One way to date non-ceramic remnants is to 
use POSL on sediments (rock fragments of size at most 2mm). Sediment contains quartz, and when buried 
underground it absorbs radiation from the soil around it. The longer the sediment is buried, the more radiation it 
absorbs. The POSL machine stimulates the radiation stored in the quartz, and then measures the photons 
escape the sediment. The older the sediment is, the more photons will be detected.
The process of dating stratigraphic layers that have non-ceramic remnants can take several months. This has 
led archaeologists to seek advanced tools for establishing layer dates in real-time, in order to make more 
informed and effective decisions regarding the progress of the excavation.

![what are sediments](https://github.com/shmooel28/finalProject/blob/master/utils/what%20are%20sediment.png)



## Dataset used
The full data set is still in development. The current available data is located [here](https://github.com/shmooel28/finalProject/blob/master/data_b1.xlsx) 


## Training
Given the relative lack of aviliable data, we created 3 different Types of models for training. To train a single model run one of the following command:

#### Decision tree model training:

    python Decision_tree.py

#### KNN model training:

    python KNN.py

#### SVM model training:

    python SVM.py
    
## Testing
for testing, we created a simple GUI interface for users. In order to activate it, run the following command:

    python GUI.py
