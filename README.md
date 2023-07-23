## Introduction
* The program will receive pre-exist parameters from the researcher and export sediment dating of the ground, using machine learning methods.

* Sediments dated by clay will be characterized by parameters of chemical and mineralogical composition, and Optically Stimulated Luminescenc (OSL) signal. The OSL signal accumulated in the soil grains during their burial. The more time that passes, the value of the signal increase.
* The program will help to quickly date, while excavating, sediments that do not have an obvious date. This is a tool with great potential to improve the fieldwork of archaeologists. 



## Dataset used
The full data set is still in development. The current available data is located [here](https://github.com/shmooel28/finalProject/blob/master/data_b1.xlsx) 


## Training
We created 3 different Types of models for training. To train a single model run one of the following command:

#### Decision tree model training:

    python Decision_tree.py

#### KNN model training:

    python KNN.py

#### SVM model training:

    python SVM.py
    
## Testing
