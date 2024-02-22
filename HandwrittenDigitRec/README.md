# Handwritten Digit Recognition

This is a program that utilizes a simple nueral network structure to analyze a set of images and provide 
a guess as to what the value displayed is. 

# Training
The training for the model can be completed in the model_training.py file. This model uses the MNIST
data set to provide the training for the model. It completes 5 epochs in the program provided but this can
be changed by editing the value 5 within the program. Layers and activation functions can also be manipulated 
according to specifications. Following this process a model name can be selected to store the generated model. 

In the case of this program a file handwrittentwo.model is used and this is also provided in the repository. 

# Execution
To run the model use the running_model.py file which will load the model you have created(ensure that
the correct model name is indicated in that file) and analyze the images that you give it. Some sample
images in the correct format have been provided in the digits folder of the repository. To analyze your 
own images, ensure that you work on an editing software that is 28 by 28 pixels, allowing the image to be
input into the program so the data can be processed correctly into the network.
