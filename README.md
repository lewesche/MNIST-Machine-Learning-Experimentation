# MNIST-Machine-Learning-Experimentation
The goal of this project was to compare the effects and characteristics of different hidden layer implementations and neural net parameters using Matlabs Machine Learning package, as well as participate in the classification challenge hosted by Kaggle. 
#
The Modified National Institute of Standards and Technology (MNIST) data set consists of pictures of handwritten numbers, 0-9, in the form of a training set and a data set. The training set contains 42,000 images of labeled digits, while the test set consists of 28,000 unlabeled images. The MNIST data set was compiled and provided by NYU, Google, and Microsoft Research Institute. 
#
A series of tests were run in which a single hidden layer was used to classify the data sets. The single layers tested included a 2D convolution layer, a rectified linear unit layer, and a max pooling layer. In the fourth test, all three of the layers were used in conjunction. Five trails were run of each test. In each trial, 90% of the training images were randomly selected and used as a preliminary training set and the remaining 10% of training data was used as a preliminary validation set. The four networks under test were evaluated by calculating their average accuracy over the five trials.
#
After the four network architectures were validated, they were retrained using the entirety of the provided training data and the unlabeled test data was classified. It was then uploaded to Kaggle and scored. 
#
The MNIST data set is available for download at:
ttp://yann.lecun.com/exdb/mnist/
Or alternativley at:
https://www.kaggle.com/c/digit-recognizer
The link above also hosts the classification challenge, where the final classification results of each network architecture under test were submitted and scored. 
#
Refer to the PDF documentation for details on methods and results. 

