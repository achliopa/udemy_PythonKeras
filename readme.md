# Udemy Course: Zero to Deep Learning with Keras and Python

* [Course Link](https://www.udemy.com/zero-to-deep-learning/learn/v4/overview)

## Section 1 - Introduction

### Lecture 2 - Real World  applications of Deep Learning

* Deep Learning > Machine Learning
* Many Industries, many dat atypes
* object recognition, caption generation
* facebook face recognition
* deep learning on speech (sound as input -> convert speech to text)
* neural machine transaltion (realt time translationsound->text-> trnaslate->sound)
* smart email reply (small feedforward network to decide if he willreply or not) -> deep recurrent nn to build the reply
* automate diagnoses of retinal disease (uses a pretrained convolutional neural network)
* diagnose skin cancer using deep neural networks
* deep learning to control cooling systems
* agriculture (monitor crops)
* cucumber sorter for farms
* self driving cars
* gaming (alphaGo) [human expert positions -> classification -> supervised learning policy network -> self play -> reinforced learning policy network -> generates new data (30mil positions -> regression -> value neteork)]
* machine generated music
* neural doodles

### Lecture 3 - Download and install Anaconda

* we have anaconda

### Lecture 4 - Installation Video Guide

* [github repo](https://github.com/dataweekends/zero_to_deep_learning_udemy)
* [tensorflow install docs](https://www.tensorflow.org/install/)
* we have tensorflow in our conda env (better choice)
* if we get stuck we will use courses conda env
* we will use it as it contains a lot of libs we dont have in our other env. 
* however installs an older tensorflow version and keras as a separate lib (we used to use it inside tensorflow)
```
cd <courseRepofolder>
conda env create
source activate ztdl
# launch jupyter
jupyter notebook
# launch browser https://localhost:8888
```
* to remove env (deactivate) while in it  stop jupyter `source deactivate`
* to delete env `conda remove -y -n ztdl --all`
* [FloydHub](www.floydhub.com) hosts GPUenabled environemtns for deep learning, it offer 100free hours 

### Lecture 6 - Course Folder Walkthrough

* we used environment.yml file to se the environment
* course folder has 10 notebooks with course material + exercises
* solutions folder provides solutions to the notebook exercises
* data folder contains datasets for the course

### Lecture 7 - First Deep Learning Model

* we import numpy and matplotlib
* we import a make cicles helper function from sklearn `from sklearn.datasets.samples_generator import make_circles`
* % for ipython magic
* we use make_curves to make cicles (ask for 1000 samples)
```
X, y = make_circles(n_samples=1000,
                    noise=0.1,
                    factor=0.2,
                    random_state=0)
```
* the first utput has shape (1000,2) it is a 2d matrix 1000by2 so 2 feats per sample. y has a class per sample 0 or 1
* we plot the data
```
plt.figure(figsize=(5, 5))
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosses")
```
* we have 2 circle shapped clusters of data one ecircling the other
* we will build a DNN to learn to separate the 2 clusters f data
* we import keras libs (keras is backend supported by tensorflow)
```
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
```
* in 4 lines we define the model
* our model is sequential with 2 layers. first uses tanh and second sigmoid
* we use SGD as optimizer (stochastic gradient descent) with alearnign rate of 0.5 and metric the accuracy it achieves
```
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
```
* in 1 line wwe fit (train) the model setting anum of epochs
```
model.fit(X, y, epochs=20)
```
* loss decreases and accuracy increases through epochs
* then we generate some horizontal and vertical ticks (linspace)
* we make them a coordinate matrix
* we make our predictions based ont eh dots
* the we plaot all. the classification works ok
```
hticks = np.linspace(-1.5, 1.5, 101)
vticks = np.linspace(-1.5, 1.5, 101)
aa, bb = np.meshgrid(hticks, vticks)
ab = np.c_[aa.ravel(), bb.ravel()]
c = model.predict(ab)
cc = c.reshape(aa.shape)

plt.figure(figsize=(5, 5))
plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosses")
```

## Section 2 - Data

### Lecture 8 - Tabular Data

* 