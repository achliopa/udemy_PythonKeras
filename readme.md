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

* table format (rows-columns)
* csv,tsv
* dbs (tables)
* a row is a record. 
* in ML a feat is an individual measurable property of something being observed. propertes to characterize our data
	* measured
	* calculated (engineered with featured enginewering)
* data can be continuous (measured) or discrete (categorical)
* not all features are informative
* a lot of emfasis is placed in feature engineering and feature selection

### Lecture 9 - Data Exploration with Pandas CodeAlong

* we usually start or ML process with data exploration
	* get quick facts
	* evident problems
	* obvious usefull feats
* In data exploration we answer the questions
	* size of dataset?
	* N features?
	* missing data?
	* data type?
	* deistribution/correlation
* pandas is THE way to data aexploration
* it reads from virtually any source
* we iport them , read the csv and see the dfs chars (head() info() describe())
* in info() we see how many non-null vals we have (find missing data)
* Pandas allows *indexing* of day in many ways
	* by index location: `df.iloc[3]` we get a row (Series object)
	* slicing indexes and select column by label: `df.loc[0:4,'Ticket']` we get multiple datapoint (indexed) (Series Object)
	* by column label: `df['Ticket']` get back a column (Series object). with double brackets we get a Dataframe
	* by multiple columns: `df[['Ticket','Embarked']]` get back multiple cols (Dataframe) .Only with double brackets
	* to column selections i can get the .head() passing in how many rows
* We can make conditional selection with pandas. the inside part returns a table of booleans that wehn passed in as datafram selection condition XORs only the tru cells. we get back a dataframe
	* `df[df['Age'] > 70]` simple conditional returns filtered df (not applied to the original df)
	* `df['Age'] > 70` returns bool ( we can count it)
	* `df.query("Age > 70")` we can filter with string queries
	* `df[(df['Age'] == 11) & (df['SibSp'] == 5)]` combined condtionals (complex) | and & used
	* `df.query('(Age == 11) | (SibSp == 5)')` using a string query
* with `.unique()` we get the unique vals in a column . usefull for categorical data to hot encode them or count them from classigfication (if labels)
* we can sort data with df.sort_values() passing the order `df.sort_values('Age', ascending = False).head()`
* we can get aggregates with specialized methods appliet o a column selection or a groupby method (chained)
	* aggregate methods .value_vounts(), min(), max(), mean(), median(), std()
	* simple example: `df['Age'].max()`
	* group by then column seletion then aggregate `df.groupby(['Pclass', 'Survived'])['PassengerId'].count()`
	* better save aggregation as a new val
	```
	std_age_by_survived = df.groupby('Survived')['Age'].std()
	std_age_by_survived
	```
	* aggregation result is a Series object to convert it back to DataFrame we  use .reset_index() and store it to a new val
* we can round up decimals with .round(decimals) passing in how many decimals we want
* we can merge two dataframs specifying the column on which we will do the merge `df3 = pd.merge(df1, df2, on='Survived')`
* we can pass new column labels to a dataframe `df3.columns = ['Survived', 'Average Age', 'Age Standard Deviation']`
* we can resuffle a datafram with the powerful pivot method
```
df.pivot_table(index='Pclass',
               columns='Survived',
               values='PassengerId',
               aggfunc='count')
```
* we can make new columns `df['IsFemale'] = df['Sex'] == 'female'` here a boolean column based on some condition
* we can genrate the correlation value of a column withe the other columns in the dataframe and sort the values (default ascending order)
```
correlated_with_survived = df.corr()['Survived'].sort_values()
correlated_with_survived
```
* 0 correlation is unrelated. positive or negative correlation is usefule
* we can use panda plot to plot the correlations
```
correlated_with_survived.iloc[:-1].plot(kind='bar',
                                        title='Titanic Passengers: correlation with survival')
```

### Lecture 10 - Visual Data Exploration

* for continuous data => lineplot
* find correlated data => scatter plot
* distributions => histogram (we lose order info)
* subtipe of histogram => cummulative distribution (s curve)
* compare distributions => boxplot

### Lecture 11 - Plotting with Matplotlib

* we create 3 datasets as numpy arrays
```
data1 = np.random.normal(0, 0.1, 1000)
data2 = np.random.normal(1, 0.4, 1000) + np.linspace(0, 1, 1000)
data3 = 2 + np.random.random(1000) * np.linspace(1, 5, 1000)
data4 = np.random.normal(3, 0.2, 1000) + 0.3 * np.sin(np.linspace(0, 20, 1000))
```
* we stack them vertically using transpose `data = np.vstack([data1, data2, data3, data4]).transpose()` to stack them as collumns (1000,4) ranther than (4,1000) if we didnt use transpose()
* we convert it to dataframe `df = pd.DataFrame(data, columns=['data1', 'data2', 'data3', 'data4'])`
* *LINEPLOT*
	* in pandas (all columns): `df.plot(title='Line plot')`
	* same line plot in matplotlib (more control):
	```
	plt.plot(df)
	plt.title('Line plot')
	plt.legend(['data1', 'data2', 'data3', 'data4'])
	```
* *SCATTERPLOT*
	* in pandas (lineplot style): `df.plot(style='.')`
	* in pandas (correlation of 2 columns):
	```
	df.plot(kind='scatter', x='data1', y='data2',
            xlim=(-1.5, 1.5), ylim=(0, 3))
	```
* *HISTOGRAM*
	* in pandas (all columns)
	```
	df.plot(kind='hist',
        bins=50,
        title='Histogram',
        alpha=0.6)
	```
* *CUMMULATIVE DISTR - HISTOGRAM
	* in pandas (s-curve) normalized:
	```
	df.plot(kind='hist',
        bins=100,
        title='Cumulative distributions',
        normed=True,
        cumulative=True,
        alpha=0.4)
    ```
* *BOXPLOT* panda style
	* in pandas(all columns): `df.plot(kind='box', title='Boxplot')`

* *SUBPLOTS* panda style
	* we first set the grid (cols,rows) as axes `fig, ax = plt.subplots(2, 2, figsize=(5, 5))`
	* then we plot passing in the subplot index (position in table) `df.plot(ax=ax[0][0],
        title='Line plot')`
    * we use tight layout so that legends and all are in place `plt.tight_layout()`
* *PIECHARTS*
	* we use aggregates to generate categorical data from continuous and count them
	```
	gt01 = df['data1'] > 0.1
	piecounts = gt01.value_counts()
	piecounts
	```
	* we do the plot (piechart panda style)
	```
	piecounts.plot(kind='pie',
               figsize=(5, 5),
               explode=[0, 0.15],
               labels=['<= 0.1', '> 0.1'],
               autopct='%1.1f%%',
               shadow=True,
               startangle=90,
               fontsize=16)
	```
* *HEXPLOT panda style*
	* we generate data (almost identical continuous data)
	```
	data = np.vstack([np.random.normal((0, 0), 2, size=(1000, 2)),
                  np.random.normal((9, 9), 3, size=(2000, 2))])
	```
	* we make them dataframe of 2 columns `df = pd.DataFrame(data, columns=['x', 'y'])`
	* we make line plot `df.plot()`
	* we make kde plot `df.plot(kind='kde')`
	* we do the hexbin plot of one column to the other (correlation) `df.plot(kind='hexbin', x='x', y='y', bins=100, cmap='rainbow')`

### Lecture 12 - Unstructured Data

* images,sound,text
* to convert unstructured data to features to be used in machine learning we fdo Feature Extraction usually with Deep Learning (NNs)
* Image is a matrix of pixes (1 or 3 color values) so a 3d table HxWxC
* usually we flatten out into a single array `(H,W,C) => (H*W*C,)`
* when we flatten out we loose corellation of a pixel to its neighbours
* what we care in feature extraction in image is not the pixel but the tegel or kernel (a small subset of image)
* these are used in CNN
* Sound is a long series of numbers. we can plot it out. x axis is time and y is the amplityde of the sound
* if we flatten it out we can stack  sounds to a table but its problematic as the size differs (not all sounds have same duration)
* sounds carry the information in frequencies . raw format (amplitude is not very useful)
* NNs can encode sound directly
* text we can extract features with words frequency(TFIDF), embeddings(WORD2VEC), parts of speech using NNs

### Lecture 13 - Images and Sound in Jupyter

* we use PILLOW library to import images
* we import Image class from pillow `from PIL import Image`
* we use the class to open an image file and get its contents `img = Image.open('../data/iss.jpg')`
* if we call `img` we render the image its of type `PIL.JpegImagePlugin.JpegImageFile`
* we transform the image to a numpy array using the .asarray() method `imgarray = np.asarray(img)`
* the type of the converted image is a numpy ndarray and the shape `imgarray.shape` is ((435, 640, 3)) so HxWxC
* we can flatten the image array into a single dimension flat array with .ravel() `imgarray.ravel()` its shape is 
(835200,) equal to HxWxC

* for Sound we use the  wavfile class from scipy.io
* we import it `from scipy.io import wavfile`
* we use it to read in a wav file `rate, snd = wavfile.read(filename='../data/sms.wav')`
* we get back 2 objects
	* snd of type an numpy.ndarray of shape (110250,) (sound length) containing the soundbits as int16
	* rate of type int which is the bitrate 44100
* we can play the sound in jupyter using the audio player. we impor it `from IPython.display import Audio` and we instantiate it passin in the sound obkects `Audio(data=snd, rate=rate)` we view a sound player
* we can plot the snd object `plt.plot(snd)` what we get is the waveform
* we can use matplotlib *scectrogram* plot passing in the NFFT param and the rate as Fs to see the sound spectrogram to see the sound frequency over time (COOL!!)

### Lecture 14- Feature Engineering

* feature engineering needs 
	* expretise
	* domain knowledge
	* diffult and expensive process
* if we train a ML model on face recognition we can extract feats by identifying key points on face and measuring distance between them. we go thus from a 2D image to an Ndistance vector
* Deeo Learning Disrupted Feature Engineering as itself finds the best way to extract the features: More powerful and faster approach
* Deep learning IS our Domain Expert
* Traditional feature engineering takes time
* deep learning is automated feature engineering

## Lecture 15 - Exercises

* easy way to split between two classes for box plotting `dfpvt = df.pivot(columns='Gender',values='Weight')`
* scatter_matrix
```
from pandas.tools.plotting import scatter_matrix # now pandas.plotting
_ = scatter_matrix(df.drop('PassengerId',axis=1),figsize=(10,10))
```

### Section 3 - Machine Learning

### Lecture 25 - Machine Learning problems

* supervised learning, unsupervised learning, reinforced learning
* deeplearnign is applied to all

### Lecture 26 - Supervised Learning

* SL helped us get rid of spam emails
* our sorting of emails to spam fed the supervised learning algos
* Binary Classification
	* churn prediction
	* sentiment analysis (on celebrity posts based on responses)
	* click prediction
	* disease screening
	* human detection (humanor bot on captchas)
* Multi class classification
* Regression (continuous)

### Lecture 27 - Linear Regression

* in linear regresion we seek a pattern a correlation between some feats an a continouous label we want to predict.
* the pattern is a linean that matches the correlation
* in linear regression we use a linear function to deduct the target val form a feature (or features)
* in multi feature datasets instead of line we have a multidimensional linear plane
* in 2d the linear regression is a ycalc=wX+b 
* w -> slpe , b -> offset

### Lecture 28 - Cost Function

* to choose the best model we should measure how googd it is
* in supervised learning we know the targets ytrue
* the residual is the offset ytrue-ycalc. it is signed
* the total error is  Σ|ei| = Σ|ytruei-ycalci| this si a cost function.
* usually we use MSE (mean square error) or RMSE
* MSE penalizes large differences (squares)
* MSE is smooth and with global minimum

### Lecture 29 - Cost Function Code Along

* we ll see how to load data, plot them and calculate the cost function
* we import matplotlib, pandas and numpy
* we import data.
* we scatter plot height vs weight and manually draw a line  of our linear fit
```
df.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults')

# Here we're plotting the red line 'by hand' with fixed values
# We'll try to learn this line with an algorithm below
plt.plot([55, 78], [75, 250], color='red', linewidth=3)
```
* we define a llinear function y
```
def line(x, w=0, b=0):
    return x * w + b
```
we set input linspace `x = np.linspace(55, 80, 100)` set w and b to 0 `yhat = line(x, w=0, b=0)` and redo the plot this time plotting th eline as yhat. (its parallel to x axiz and 0)
* we calcualte MSE in a function
* we use it to ge the error (we use flattening to y_pred)
```
def mean_squared_error(y_true, y_pred):
    s = (y_true - y_pred)**2
    return s.mean()
X = df[['Height']].values
y_true = df['Weight'].values
y_pred = line(X)
mean_squared_error(y_true, y_pred.ravel())
```
* with df['column'].values we get a numpy array froma pandas series object

### Lecture 30 - Finding The best Parameter

* choose b and w (small amount) so that MSE decreases... then it goes to a minimum and starts increasing again
* its a param - cost plot that U shaped. param is not an axis but a plane... its a 3 3 Cup (concave) shaped plot. in its bottom are the b,w vals that give miniimum cost
* Finding the best combination is called TRAINING
* training => minimize cost
* minumum cost => best model

### Lecture 31 - Linear Regresion COde Along
* we plot a scatterplot of weight vs heignt
```
ax1 = plt.subplot(121)
df.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults', ax=ax1)
```
* we make an array of biases
```
bbs = np.array([-100, -50, 0, 50, 100, 150])
```
* we set w constant =2 and calcculate MSE and plot the linear fit for each w,b combo
```
for b in bbs:
    y_pred = line(X, w=2, b=b)
    mse = mean_squared_error(y_true, y_pred)
    mses.append(mse)
    plt.plot(X, y_pred)
```
* we plot the sost vs b curve
```
# second plot: Cost function
ax2 = plt.subplot(122)
plt.plot(bbs, mses, 'o-')
plt.title('Cost as a function of b')
plt.xlabel('b')
```
* we see the minimum between 0 and 50
* we ll go Linear Regression with Keras
* we import the type of model we want to implement (Sequential) `from keras.models import Sequential`
* we import the type of layers we want to use (Dense) `from keras.layers import Dense`
* we import our optimizers `from keras.optimizers import Adam, SGD`
* we create a model instance `model = Sequential()`
* we add a layer to our model `model.add(Dense(1, input_shape=(1,)))` we use 1 neuron (1 output) and our input is a 1d array of size 1
* we print out the model summary `model.summary()`
* the output print is
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1)                 2         
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
```
* we have 2 params (w,b) 1 output shaped (None,1) none is reserved for batches
* in keras we need to compile the model `model.compile(Adam(lr=0.8), 'mean_squared_error')` or else constuct the model using the backend (in our case tensorflow)
* for the same model we can use various bakcends
* in compile we define the cost function we want to use
* next we train our model  passing input and output placeholders and the num of epochs `model.fit(X, y_true, epochs=40)`
* loss is dropping
* we can now use our model to do predictions `y_pred = model.predict(X)`
* we now plot our data set (X,y_true) points and the y_pred as a line (linear fit)
```
df.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults')
plt.plot(X, y_pred, color='red')
```
* we print out W,b using  model.get_weights() method `W, B = model.get_weights()`

### Lecture 32 - Evaluating performance

* we ll learn how to define a baseline model, ho to use score to compare models, hot to  do train/test split
* in ou previous model we minimzed cost but we worked only with our dataset. we dont know how our model will perform in anoher dataset or with data
* we need a simple model to use as areference
* we need a score to compare different models. (the cost cannot do the job as it is dependent on value scale)
* a common score for regression models is R2. the R2 compares the sum of the squares of residuals in our model with the sum of the squares of a baseline model that predicts the average price all the time R^2 = 1 -(SSres/SStot)
* In a ggod model the R^2 is close to 1
* R^2 can have negative val if our  model is less than average
* to check if our model generalizes well we need t split our data to training and test set

### Lecture 33 - Evaluate Performance Code Along

* we import train test split `from sklearn.model_selection import train_test_split`
* we split the data `X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2)`
* we reset the weights of our model 
```
W[0, 0] = 0.0
B[0] = 0.0
model.set_weights((W, B))
```
* we detrain the model by choosing to reset W and B to retrain it using the train set `model.fit(X_train, y_train, epochs=50, verbose=0)`
* we use the x_train data to make our predictions `y_train_pred = model.predict(X_train).ravel()` note we flatten in the input data (weird as we trained on it)
* we do the same for test data `y_test_pred = model.predict(X_test).ravel()`
* we import mse from sklearn
* we calculate and print it out 
```
print("The Mean Squared Error on the Train set is:\t{:0.1f}".format(mse(y_train, y_train_pred)))
print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y_test, y_test_pred)))
```
* we see that MSE is similar between test and train which is great
* we calculate R2
```
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
```
* R2 is good and similar

###  Lecture 34 - Classification

* example problem. if customer will buy or not based on time spent on site
* classification is supervised learning on discrete targets (categorical  data)
* logistic regression is a typical classification model
* if we scaterplot linear vs cat3egorigal data is not uuseful
* the logistic regression regression predicts modeling the probability of the outcome using the logistic curve
* ith ehypotheses uses the sigmoid function ypred = sigm(b+Xw)
* the cost function cannot be the MSE. for claasification we use log loss or cross entropy cost
* the classification cost is calculated from ci = -(1-yi)log(1-ypredi)-yilog(ypredi) y and yrped can be 0 or 1
* ci = -log(1-ypredi) when yi =0 or -log(ypredi) when yi = 1 
* when y = 1 we expect ypredi to be 1 (in sigmoid this is when x is large and possitive) so we make cost small
* if x is negative we make cost larger and larger. 
* when y is 0 we expect ypred to be 0 and x is pushed to negative val
* we have the cost for an individual point i
* the general cost is c = (1/N)Σici (average cross entropy) or binary log loss
* to select the best params that minimize thhe cost. logistic regression calculates probability. we need to convert it to a binary predictions. the simplest way is to set a threshold e.g P > 0.5 => ypred = 1 otherwise y = 0
* a metric for evaluating our classification model is accuracy (number of correct predictions / total predictions)
* usually we compare accuracy in training set with that on test set to see how well our model generalizes

### LEcture 35 - Classification Code Along

* we load our dataset from csv
* we inspect it. just 2 columns one is target (labels) 1 feat time on site (cont)
* we do a scatterplot `df.plot(kind='scatter', x='Time (min)', y='Buy')`
* we instantiate our KERAS model (sequential) and add 1 dense layer of 1 neuron (1 output), 1 input and set activation method as sigmoid (in linear regression we did not set an activation merthod)
```
model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
```
* we compile our keras model passing  in params (binary cross entropy as cost method, SGD 'gradient descent' as optimizer, accuracy as metric) `model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])`
* we print out the summary
* we set our input and output arrays (from pandads to numpy) and train our model with them (no split) setting 25 epochs
```
X = df[['Time (min)']].values
y = df['Buy'].values

model.fit(X, y, epochs=25)
```
* our accuracy is not very good (not many data)
* we plot our data as scatterplot and plot the sigmoid passing in linspaced data nd using `model.predict(temp)` to cslculste the y val (probability)
```
ax = df.plot(kind='scatter', x='Time (min)', y='Buy',
             title='Purchase behavior VS time spent on site')

temp = np.linspace(0, 4)
ax.plot(temp, model.predict(temp), color='orange')
plt.legend(['model', 'data'])
```
* we use a threshold to go from probabilities to  binary data (binary array) `temp_class = model.predict(temp) > 0.5`
we plot the thresholded data as matplotlib treats booleans as 0 or 1 
```
temp = np.linspace(0, 4)
ax.plot(temp, temp_class, color='orange')
plt.legend(['model', 'data'])
```
* we get the model accuracy by getting the predictions and converting it to binsry using teh threshold
```
y_pred = model.predict(X)
y_class_pred = y_pred > 0.5
```
* we import the metric and print it out passing the label (classes) of the y_true and the predictions in binary format `print("The accuracy score is {:0.3f}".format(accuracy_score(y, y_class_pred)))`
* we ll now do the train test split train on train data and evaluate both with train and test 
* we split our dataset 
```
params = model.get_weights()
params = [np.zeros(w.shape) for w in params]
model.set_weights(params)
```
* we get the model weights and zero them out
```
params = model.get_weights()
params = [np.zeros(w.shape) for w in params]
model.set_weights(params)
```
* we print out the accuracy `print("The accuracy score is {:0.3f}".format(accuracy_score(y, model.predict(X) > 0.5)))` it is 0.5 which is expected for an untrained model
* we train the model with train data `model.fit(X_train, y_train, epochs=25, verbose=0)`
* we print out accuracy both for train and test set
```
print("The train accuracy score is {:0.3f}".format(accuracy_score(y_train, model.predict(X_train) > 0.5)))
print("The test accuracy score is {:0.3f}".format(accuracy_score(y_test, model.predict(X_test) > 0.5)))
```

### Lecture 36 - Overfitting

* sigmoid function is non linear
* weigths w is a size M vector
* inputs X is a NxM matrix (N: num of records, M num of feats)
* overfitting happens when the model learns the probability distribution of the training set too well so cannot generalize to test set equally well
* If Train score >> Test score: Overfitting
* How to avoid:
	* split well. preserve labels ration
	* randomly sample dataset
	* test set not small
	* train set not small
	* reduce complexity of NN or do regularization

### Lecture 37 - Cross Validation

* do various train/test splits evaluate them and average scores
* K-form cross validation:
	* the dataset is split in K equally sized - random sampled subsets
	* we do K rounds of train-test. in each one one subset is used as test set and the rest as train sets
	* Accuragy = Average(Round Accuracy)
	* Train More - get better estimation
	* totally parallel problem (we can harness GPUs)
* Stratified Kfold is similar to kfold but it makes sure the ratios of labels are preserved in the folds
* LOLO & LPLO (leave one label out or leave p lables out). it si used if we have second label that is not target but might affect our accuracy. we leave one out or multiple out for testing to simulate a new unknown set of data (use groupby)

### Lecture 38 - Cross Validation Code Along

* we import a wrapper to use keras models in scikit learn `from keras.wrappers.scikit_learn import KerasClassifier` as cross validation function is in sccikit learn
* we implement a helper function that does the whole keras model building sequence
```
def build_logistic_regression_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(SGD(lr=0.5),
                  'binary_crossentropy',
                  metrics=['accuracy'])
    return model
```
* we do this because the KerasClassifier needs our model as a function return
```
model = KerasClassifier(build_fn=build_logistic_regression_model,
                        epochs=25,
                        verbose=0)
```
* having our model in scikit learn allows us to use its tools on it. we import cross validation and KFold `from sklearn.model_selection import cross_val_score, KFold`
* we make an cross validation instance doing 3 folds `cv = KFold(3, shuffle=True)`
* to get the scores we use crossvalscore passing the model and the cross val method `scores = cross_val_score(model, X, y, cv=cv)`
* we print out our scores from 3 folds `scores`
* and calculate tyhe mean and std `scores.mean(), scores.std()`

### Lecture 39 - Confusion Matrix

* accuracy to judge our classsification model gives an indication about how we perform overall
* the confusion matrix gives more insight TP,TN,FP (Type I error),FN (Type II Error)
* Accuraty (TP+TN)/total
* Precision TP/ (TP+FP)
* Recall TP/(TP+FN)
* FI = 2PR(P+R)
* these metrics scale for categorical data in more classes

### Lecture 40 - Confusion matrix Code Along

* we import it from scikit `from sklearn.metrics import confusion_matrix` and print it out `confusion_matrix(y, y_class_pred)`
* we can print it out as panda dataframe (prettyfy)
```
def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted '+ l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df
pretty_confusion_matrix(y, y_class_pred, ['Not Buy', 'Buy'])
```
* we import rest of classification metrics
```
from sklearn.metrics import precision_score, recall_score, f1_score
```
* we print them out
```
print("Precision:\t{:0.3f}".format(precision_score(y, y_class_pred)))
print("Recall:  \t{:0.3f}".format(recall_score(y, y_class_pred)))
print("F1 Score:\t{:0.3f}".format(f1_score(y, y_class_pred)))
```
* easiest way to print the classification report `from sklearn.metrics import classification_report`
```
print(classification_report(y, y_class_pred))
```

### Lecture 41 - Feature Preprocessing Code Along

* we ll use onehot encoding for categorical classes 
* we read out data `df = pd.read_csv('../data/weight-height.csv')` and view it with .head()
* we check unique vals in gender `df['Gender'].unique()`
* we make dummy cols (one hot encoding) these are mututally exclusive `pd.get_dummies(df['Gender'], prefix='Gender').head()` we get a new col per unique val
* ANNs work best with feats scaled to 0-1 vals (due to activation fucntions used) so we rescale
* we can rescale manually
```
df['Height (feet)'] = df['Height']/12.0
df['Weight (100 lbs)'] = df['Weight']/100.0
```
* we get tranges comparable to 1 but not 1 `df.describe().round(2)`
* to scale in 0-1 range we use MinMaxScaler `from sklearn.preprocessing import MinMaxScaler`
* we instantiate it and call fit_transform
```
ss = StandardScaler()
df['Weight_ss'] = ss.fit_transform(df[['Weight']])
df['Height_ss'] = ss.fit_transform(df[['Height']])
df.describe().round(2)
```
* we can isntead do standard normalization (mean of data is 0 and std deviation is 1) hoowever outliners will be out of 0-1 range
```
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df['Weight_ss'] = ss.fit_transform(df[['Weight']])
df['Height_ss'] = ss.fit_transform(df[['Height']])
df.describe().round(2)
```
* we plot the histogram of the 2 normalization resutls
```
plt.figure(figsize=(15, 5))

for i, feature in enumerate(['Height', 'Height (feet)', 'Height_mms', 'Height_ss']):
    plt.subplot(1, 4, i+1)
    df[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)
```
* plots are same but only ranges vary

### Lecture 42 - Exercise 1 

* need to convert dataframes to numpy.ndarrays prior to KEras insertion (use .values)

### Lecture 44 - Exercise 2 

* benchmark is the easiest model we can build `df.left.value_counts()/len(df)` 0 is 76% so predicting everybody stayes will give 76% accuracy
* multiple column dummies `df_dummies = pd.get_dummies(df[['sales','salary']])`

## Section 4 

### Lecture 46 - Deep Learning Successes

* difference betweeen deep learning and traditional machine learning
	* deep learning models get better (better performance) as we throw more training data inthem , ML improves and plateaus
	* Easier pipeline: DL => Input -> DL -> Output, ML => Input -> Feature Extraction -> Feats -> Shallow ML algo -> Output

### Lecture 47 - Neural Networks

* Neuron: Multiple Inputs -> One Output (One activation function)
* Linear regression as NN: b-> , x(W)-> [xW+B] -> y=xW+B (neuron does one mul and one add)
	* if we have multiple inputs and weights x and W are vectors so the multiplication is . (dot) the notation is the same.. y = xdotW+b
* to go from linear regression to logistic we just apply a sigmoid  activation function notation is same but after neuron -> sigmoid -> y = sigmoid(xdotW+b)
* the first Neural network invented (Perceptron) is exactly the same but with a different activation function y = H(xdotW+b) H is a step function (0 if input < 0 1 if input > 0) it has a discontinuity at 0 so bias is not 0

### Lecture 48 - Deeper networks

* Deep networks have hidden layes . multiple neurons per layer and multiple layers
* we build deep nns by stacking perceptront one to the other
* say we have vector input of size N and a layer of N neurons, 1 per vector element.mathematically is represented Z1 = XdotW1+B1 = Σj(xijW1jk)+bj for every node k. this is layer 1
* activation is appied afte input is weighted and biased O1 = actf(Z1)
* second layer does the same O2 = actf(O1dotW2+B2)
* this is a fully connected multilayer network

### Lecture 49 - Neural network Code along

* we import the libraries
```
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```
* we import from sklearn moon shape data generator
```
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.legend(['0', '1'])
```
* the data are 2 classes of interleaping moons . the separation line should be a curve not a line
* we do the train test split
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
* we import keras libs
```
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
```
* we implement a shallow model of logistic regression with 1 layer
```
model = Sequential()
model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
```
* we train it with train data for 200 epochs `model.fit(X_train, y_train, epochs=200, verbose=0)`
* we get the results `results = model.evaluate(X_test, y_test)` and print them out. second element that is accuracy. the accuracy is 0.85
* we printout the decision boundary. it is a straight line
```
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    c = model.predict(ab)
    cc = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
    plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    plt.legend(['0', '1'])
    
plot_decision_boundary(model, X, y)
```
* we build a dense model of 3 layers (4 neurons in first, 2 in second 1 in last)
```
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
```
* we train it `model.fit(X_train, y_train, epochs=100, verbose=0)`
* we get the results `model.evaluate(X_test, y_test)` accuracy is 1 (excelent)
* we get accuracy for train and test data
```
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

print("The Accuracy score on the Train set is:\t{:0.3f}".format(accuracy_score(y_train, y_train_pred)))
print("The Accuracy score on the Test set is:\t{:0.3f}".format(accuracy_score(y_test, y_test_pred)))
```
* and plot the boundary. it is a nicely fitted curve

### Lecture 50 - Multiple Outputs

* we will learn to extend our network for multiple outputs
* this applies to multiclass classification problems (A<B<C)
* or regression problems where the output is a vector of values (self driving cars=> predict direction and speed of car)
* we add as many output nodes as the elements in the vector or the classes in our multiclass classification problem
* each node will generate an independent value that gets assigned to a componenbt of the vector
* in classification we can have mutually exclusive classes (A,b or C) or non exclusive classes (e.g tags)
* in both cases we generate binary dummy columns (in mutually exclusive we can have only one 1, in non exclusive we can have more 1s)
* for independent tags each one needs to be noprmalized to 0 -1. we use a sigmoid activation function on each output node as probabilities are independent
* for mutually exclusive classes. we need an activation function that forces the sum of the outputs to be 1. we use SOFTMAX. σ(z)j = exp(zj)/Σk=1toK(exp(zk)) for j=1,..,K
* vector regression => many output nodes
* multi class classification => softmax output
* non exclusive tags => many sigmoid output nodes

### Lecture 51 - Multiclass Classification Code Along

* we load the iris dataset `df = pd.read_csv('../data/iris.csv')`
* we do a seaborn pairplot of the numeric columns to see correlations with hue on tager class
```
import seaborn as sns
sns.pairplot(df, hue="species")
```
* we have 4 feats and 3 outputs (mutually exclusize aka softmax)
* we make X matrix (dataframe) excluding target `X = df.drop('species', axis=1)`
* we ptrint out different classes `target_names = df['species'].unique()`
* we create a dict fromn classes to assign numeric vals `target_dict = {n:i for i, n in enumerate(target_names)}`
* we make our target array using as map the dictionary of classes `y= df['species'].map(target_dict)`
* we use keras utility to make dummy hotencoded columsn for targets from our target array
```
from keras.utils.np_utils import to_categorical
y_cat = to_categorical(y)
```
* we do the train test split. `X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat,
                                                    test_size=0.2)`
* we implement our model (1 layer, 3 nodes aka outputs, 4 input vector, softmax) we use categrorical crossentropy not binary
```
model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='softmax'))
model.compile(Adam(lr=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
* we train our model `model.fit(X_train, y_train, epochs=20, validation_split=0.1)` we use validation split taking 10% of training data for validatioon to check loss
* we get our predictions `y_pred = model.predict(X_test)`
* each prediction has 3 numbers (the probability of each class) we keep the one with maximum probability
```
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
```
* we print out metrics (classification reprot and confusion matrix)

### Lecture 52 - Activation Functions

* Sigmoid in python (binary classification)
```
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
```
* sigmoid smooths all real axis to a val between 0 to 1
* Step function in python (original perceptron)
```
def step(x):
	return x > 0
```
* it has a discontinuous transition in 0 
* Hyperbolic Tangient (Tanh) in python
```
np.tanh(x)
```
* like sigmoid but goes from -1 to 1 penalizes negative values of x
* Rectified Linear Unit (RELU) in python
```
def relu(x):
	return x * (X > 0) # y=max(0,x)
```
* originates in biology. more effective than tanh  or sigmoid in NNs. it imroves training speed
* SOFTPLUS in puthon
```
def softplus(x):
	return np.log1p(np.exp(x)) #y=log(1+exp(x))
```
* is a smooth aproximation of RELU (smooth trransition in 0)
* any activation function makes the NN non-linear
* The secret power of NNs is the non-linearity between each layer allows them to deal with very complex data

### Lecture 53 - Feed Forward

* we can think of a NN as a black box that takes an imnput and applies a calculation F: y=F(x)
* this calculation is called feed-forward ( amix of linear and non-linear steps)
* input is a matrix PxN (P: num of rows => num of points, N: num of columns => num of features)
* inputs are passed to nodes by beeing multiplied by weights. Weights in first layer W1: (NxM matrix) (N: num of rows => num of features, M: num of columns => num of nodes in layer 1)
* Biases in first layer are a vector size M (M: num of nodes in layer 1)
* 1st layer performs the linear transformation O1 = XdotW1+B1 that is followed by the non-linear transformation by the activation function Z1 = sigmoid(O1). Z1 is the output of layer 1 and is a matrix (PxM)
* for layer 2 the procedure repeats. this time the input is PxM (Layer 1 output Z1) and the M for this layer is equal to the layer 2 nodes. same goes for other layers to
* this is why we can think of the whole network as a single function F
* last layer will have as many nodes as the values we try to predict

### Lecture 54 - Exercise 1

* if we do `_ = df.hist(figsize-(12,10))`
* classes overlap
* we can use seaborn heatmap to see correlations `sns.heatmap(df.corr(), annot=True)`
* we do `df.describe()` to decide on normalization of feat data. each feat has its own scale.. he uses StandardScaler and from keras the to categorical util (for hot encoding)
```
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
```
* we perform normalization and ho encoding (we should fir only on train data and do not hotencode)
```
sc = StandardScaler()
X = sc.fit_transform(df.drop('Outcome',axis=1))
y = df['Outcome'].values
y_cat = to_categorical(y)
```

## Section 5

### Lecture 62 - Derivatives and Gradient

* derivative is the rate of change of a param dx(t)/dt
* derivative is the slope of the curve rf/dt(ti) = (f(ti) - f(ti-1))/(ti - ti-1)
* partial derivatives multivariant function elevation = f(long,lat) or y = f(x1,x2) patial derivatives δf/δx1 δf/δx1
* if we are in the top of a hill the fastest way down will be in the direction the hill is more steep
* in a 2d plane of x1 and x2 the direction of the more abrupt change will be a 3d vector whose components are the partial derivatives with respect to each variable. we call this vector gradient (inverted triangle) called del
* del f = [[δf/δx1],[δf/δx2]]
* a gradient is an operation that takes a function of multiple variables and returns a vector. the components of this vector are all the partial derivatives of the function. partial derivs are functrions of all variables the gradient to is also a function of all variables (a vector function)
* as we said the gradient gives the vector of maximum steepnes to the plane of x1,x2. if we want to go downhill we tkae the negative gradient. we will use it to minimize cost function

### Lecture 63 - Backpropagation Intuition

* say we have a function of 1 variable w, the function output is on the vertical axis
* we now the function only arount pointw. we wnat to increase the output. we have to use functions slopw (derivative) to decide: w- δf/δw
* the way of looking for the minimum of the function using derivatives (in case of multiple vars the gradient) is called gradient descent
* we use the gradient descent to miniomize cost: w -> w-δcost/δw (we use this rule to update the param)

### Lecture 64 - Chain Rule

* we do gradient descent for a simple NN. one imput/one output , 2 layers of 1 node, sigmoid activation
* output of layer 1 : z1 = xw1+b1 => a1=σ(z1)
* output of layer 2 : z2 = a1w2+b2 => y^ = a2 = σ(z2)
* the cost function can be abstracted as J=J(y,y^(w,b,x)) the predicted output y^ is a function of all inputs and weights and biases
* if we calculate the partial derivative of the cost function to the layer 2 weight: δJ/δw2=δJ(y,s(z2(w2)))/δw2 where z2=a1w2+b2, y^=a2=σ(z2)
* what we did is to calculate the derivative of a nested function using the CHAIN RULE
* an example of the chain rule: h(x)=log(2+cos(x))
	* f(g)=log(g)
	* g(x)=2+cos(x)
	* h(x)=f(g(x))
	* CHAIN RULE: dh/dx = (df/dg)(dg/dx) : dlog(g)/dg = 1/g , d(2+cos(x))/dx= -sin(x) => d(log(2+cos(x)))/dx = -sin(x)/(2+cos(x))

### Lecture 65 - Derivative Calculation

* we ll calculate weight correction for a simple network and undertand why its called back propagation
* we use the previous lecture simple network and its abstract methods. we want to calculate the derivative of the cost in respect to weight of layer 2 (partial derivative. 
* as we saw before it is  δJ/δw2=δJ(y,σ(z2(w2)))/δw2 ιf we consider the layer methods (where z2=a1w2+b2, y^=a2=σ(z2)) and the chain rule it becomes (δJ/δa2)(δa2/δz2)(δz2/δw2)
	* the first term (δJ/δa2) depends on the form of the cost (δJ/δυ^) it can be calculated from the MSE, Log loss etc
	* the second term δa2/δz2 is the derivative of the activation function a2 = σ(z2) => δσ(z2)/δz2=σ(z2)σ(1-z2) using the derivative of the sigmoid
	* the third term δz2/δw2 is a derivative of alinear function z2 = a1w2+b2 so is δz2/δw2=a1
* so the δJ/δw2=δJ/δy^σ(z2)σ(1-z2)a1 this is the value we will subtract from w2 to make the rule of gradient descent w2-δJ/δw2 to update the param to minimize cost function. if we represent αλλ the layer2 contributiont δJ/δy^σ(z2)σ(1-z2)=δ2 we see that it becomes δ2a1 so is propoirtional to th layer2 input by a factor δ2
* δ2 is calculated taking into account parts of the network that are downstream to the output in respect to w2 and equals δ2=δJ/δz2 (remember chain rule)
* We can use the same approach (chain rule) to calculate the correction to the weight in the first layer w1 δJ/dw1=(δJ/δa2)(δa2/δz2)(δz2/δa1)(δa1/δz1)(δz1/δw1)=δ1x
* δ1 is proportional to δ2 : δ1=δ2w2σ(z1)σ(1-z1) thats why the method is called back propagation as we start from output and move to input calculating the correction to the weights to minimize the cost using the gradient descent

### Lecture 66 - Fully Connected Backpropagation

* we calculated the backpropagation for a simple network of only 2 nodes in series
* we ll try now to expnad to a Fully Connected network
* a fully connected network contains multiple nodes in each layer and each layer is connected to all nodes in teh previous and next layer
* weights in each node are identified by two indices kj (k: receiving node, j: emmiting node)
* the input sum at layer l and node k is zk@l = Σj(wkj@l)(aj@l-1)+b ktol so th input to activation function  is the weighted sum of all activations at layer l-1  + the bias terms at layer l
* like before we can expand the derivative of cost function respective to the weight wkj in layer l using the chain rule δJ/δwkj@l=(δJ/δak@l)(δak@l/δzk@l)(δzk@l/δwk@l) the last 2 terms (δak@l/δzk@l)(δzk@l/δwk@l)=σ'(zk@l)aj@l-1 so the derivative of the sigmoid and the activation of the previous layer. the differentiating factor is the derivative of the cost (δJ/δak@l)=Σm((δJ/δzm@l+1)(δzm@l+1/δak@l))=Σm(δm@l+1wmk@l+1)
* we see that deltas are connected by a recursive formula δk@l=(Σmδm@l+1wmk@l+1)σ'(zk@l) so we can u[date the deltas at layer l as a function of deltas at layer l+1]
* that is why the method is called backpropagation we calculate deltas at output and from them we calculate deltas at previous layers

### Lecture 67 - Matrix notation

* we will calculate backpropagation using matrices
* we rearrange all weights in a big matrix W, and all deltas at layer l in a vector called δl
* with these the dum at each Node becomes a matrix multiplication
* (W@l+1)Td@l+1=[[w11@l+1,...,wm1@l+1,...,wM1@l+1],...,[w1k@l+1,...,wmk@l+1,...,wMk@l+1],...,[w1K@l+1,...,wmK@l+1,...,wMK@l+1]]dot[δ1@l+1,...,δm@l+1,...,δM@l+1] = δ@l
* the generalized simplified formula is (for the last layer) δL = del aL J (hadamard product) σ'(zL)
* the formula in the inner layers is recursive δl=(Wl+1)Τdotδl+1(hadamard product)σ'(zl) the corrections to weights and biases are obtained by the deltas
* Backpropagation Algorithm StepByStep
	* Forwards propagation y^
	* Error in last layer δL = del aL J (hadamard product) σ'(zL)
	* Backward propagation (calculate error signals at each neuron in each layer) δl=(Wl+1)Τdotδl+1(hadamard product)σ'(zl)
	* Calculate the derivative of the cost function with respect to the weights δJ/δwkl@l=δk@l x aj@l-1. result is a matrix same size as weight matrix
	* Calculate the derivative of the cost function with respect to the biases δJ/δbj@l=δj@l (a column vector at each layer)
	* we update each weight and each bias using gradient descent rule w -> w - δJ/δw , b -> b - δJ/δb

### Lecture 68 - Numpy Arrays Code Along

* numpy fast tutorial 
* we import numpy, pandas, matplotlib
* we create an array `a = np.array([1,2,3,4])`
* we check it with `a` => array([1, 3, 2, 4])
* we check its type `type(a)` => numpy.ndarray
* we create 2D matrices of various size and check their shape
```
A = np.array([[3, 1, 2],
              [2, 3, 4]])

B = np.array([[0, 1],
              [2, 3],
              [4, 5]])

C = np.array([[0, 1],
              [2, 3],
              [4, 5],
              [0, 1],
              [2, 3],
              [4, 5]])

print("A is a {} matrix".format(A.shape)) # A is a (2, 3) matrix
print("B is a {} matrix".format(B.shape)) # B is a (3, 2) matrix
print("C is a {} matrix".format(C.shape)) # C is a (6, 2) matrix
```
* we see various ways to acces elements in array or nested arrays
```
A[0] # array([3, 1, 2])
C[2, 0] # 4
B[:, 0] # array([0, 2, 4])
```
* we can do elementwise operations in numpy (aplly operator to each element)
```
3 * A # multiply each eleemnt with 3
A - A # subtruct each eleemnt from its self (we expect all 0)
```
* we cannot do elementwise operations with arrays of different shape
* what we can do is dot product (for arrays of different shape). it is matrix mutliplicationsso matrix multiplication rules apply (we can multily 2x3 dot 3x1 but not 2x3 dot 1x3)
```
A.dot(B) # or np.dot(A,B)
# array([[10, 16],
#       [22, 31]])
```
* A shape is (2,3) B is (3,2) so B.dot(A) is (3,3)

### Lecture 69 - Learining Rate

* when we update our products in gradient descent `w -> w - δcost/δw` we move by a quantity determined by the rate of change of the cost with respect to the weight w. 
* this can be a problem in 2 ways. if the cost function is very flat we will move very slowly, if the cost function is very steep we might jump over the minimum
* we intruduce a factor α that allows us to decide how big step to take in the direction of the gradient [w -> w - αδcost/δw] this is called learning rate. small learning rate means tiny steps. big learning rate means big steps
* if learning rate is too large we might miss the solution (overshoot)

### Lecture 70 - Learning Rate Code Along

* we ll build a classifier to distinguish bank notes
* we load the dataset `df = pd.read_csv('../data/banknotes.csv')`
* check the feats `df.head()` we have 4 numeric feats and 1 target class (binary)
* we display the df using seaborn pairplot to look at correlations `sns.pairplot(df, hue="class")`
* the correlations look quite good in respect to class. (there are differences)
* we check our class count in the df `df['class'].value_counts()`
* we build our baseline model using TML (traditional machine learning) with the Random Forest Algo from sklearn
* we do our imports
```
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
```
* we make our data for the algorithm scaling them to 0-1 using scale (does not fit the data, just scales)
```
X = scale(df.drop('class', axis=1).values)
y = df['class'].values
```
* we do a default model and train it doing 3fold cross validation (default)
```
model = RandomForestClassifier()
cross_val_score(model, X, y)
```
* scores are exceptioonal
* we try logistic regression using keras 
* we split our data
```
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
```
* import our keras building blocks
```
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
```
* we import backenf (tensorflow) as we want to clear the tensorflow session to run multiple models
* build our model -> compile it -> fit it -> evaluieate it
```
K.clear_session()

model = Sequential()
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train)
result = model.evaluate(X_test, y_test)
```
* we have saved the results of fitting to a variable as we want to plot it and compare models
* we convert it to a dataframe `historydf = pd.DataFrame(history.history, index=history.epoch)`
* inspect it `historydf.head()` it has 2 cols accuracy and loss over epochs
* we plot it
```
historydf.plot(ylim=(0,1))
plt.title("Test accuracy: {:3.1f} %".format(result[1]*100), fontsize=15)
```
* accuracy is good not perfect
* we change the batchsize and learning rate
* we make an empty list for dataframes as we will build different training histories to plot for different learning rates
```
dflist = []

learning_rates = [0.01, 0.05, 0.1, 0.5]

for lr in learning_rates:

    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=lr),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, batch_size=16, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
```
* we concat dflist to a dataframe along columns
```
historydf = pd.concat(dflist, axis=1)
```
* we add multindex adding one more index level
```
etrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([learning_rates, metrics_reported],
                                 names=['learning_rate', 'metric'])

historydf.columns = idx
```
* we plot all columns using subplots
```
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()
```

### Lecture 71 - Gradient Descent

* the gradient is calculated from the cost function andevaluated on the training data
* w -> w - δJ/δw , J = J(y,y^(w,b,x)) x and y indicate a pair of training features and labels
* we could feed the training data one point at the time to the cost function and for each pair calcualte the new weights. this is called Stochastic Gradient Descent (SGD)
* once the model has sessn the training data once we say an epoch is completed and we start again
* SGD is a very noisy estimation of the gradient as a single data point is used to estimate the gradient
* We can imrove by averaging the gradient over the training data before updating the weights (Normal Gradient Descent)
* In a Normal or Batch Gradient Descent we first calculate the gradient for all training data, then we averageit to update the weights. although more accurate its not an optimal method (we do one weight update per epoch)
* An in-between solution is called MINI BATCH gradient descent. we still average the gradient over a small number of points (batch) taken from a sample of the training set (16,32,64,128,256)

### Lecture 72 - Gradient Descent Code Along

* we ll now see the effect of channging the batch size
* we do thesame multy train session like in learnign rate code along. but now in each run we pass in a different batch size
```
dflist = []

batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, batch_size=batch_size, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
```
* concat lists into df and add a new level of index (multiindex)
```
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([batch_sizes, metrics_reported],
                                 names=['batch_size', 'metric'])
historydf.columns = idx
```
* we plot in subplot per metric
```
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()
```
* best results we get from batch size = 16

### Lecture 73 - EWMA (Exponential Weighted Moving Average)

* One of the most imortant algorithms with application everywher (Finance, DSP, NNs})
* say we have a sequence of ordered datapoints (values of a stock, sensor data) anything measured in a sequence
* if the data is noisy we would like to smooth the data with a smoothing technique
* a way to remove noise from a time series is to apply a moving average, we wait till we have enough samples and use their average as an estimation of the current val
* to implement this method we need to keep many data in a buffer and update them anytime new data comes in
* EWMA solved th eproblem with a recursive formula st = axt+(l-a)st-1. we need to keep track of only the last value of the average itself
* the formula says that the moving average at time t is a mix between the value of the raw signal (xt) and the previous value of the moving average (st-1). 
* the real mixing is controlled by a param s that takes values from 0 to 1 (0% to 100%). if a is small (10%) mpst of the contribution comes from previous values of the signal. if a is big most of the contribution comes from the raw signal

### Lecture 74 - Optimizers

* optimizers = cost minimization algorithms
* so far we ve seen SGD
* an imporvent on SGD is to add Momentum (we accumulate the gradient correction in a variable called velocity which serves as a smooth version of the gradient)
* SGD with Nesterov momentum is like moementum as it accumulates gradients ina velocity variable but these variables are calculated using an interim update of the params. we perform an temprary update of the param using the previously calculated velocity, then we calculate gradients in this interim point, finally we update the params for real
* SGD and SGD Momentum keep the learnign rate constant
* Adaptive algotrithms adapt the learning rate to the size of the gradient
* AdaGrad algorithm acccumulates the square of the gradient into a variable and computes the update with the inverse function of such square. this keeps the size of the parameter update stable despite of the gradient size
* RMSProp is also adaptive but allows to choose the fraction of the square gradients to accumulate using an EWMA accumulation formula
* The ADAM Algorithm is also adaptive and uses EWMA for both the gradient and square ofd the gradient
* There is no best choice each of them performs best depending on the problem also changing hyper params

### Lecture 75 - Optimizers Code Along

* we ll see how to build different optimizers in keras
* first we import them `from keras.optimizers import SGD, Adam, Adagrad, RMSprop`
* we will run multiple training sessions like before storing all the training results (epoch wise) into an array which we will make a dataframe and plot to see the resutls. we test multiple optimizers with same learningrate but playing with hyper-params
```
dflist = []

optimizers = ['SGD(lr=0.01)',
              'SGD(lr=0.01, momentum=0.3)',
              'SGD(lr=0.01, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.01)',
              'Adagrad(lr=0.01)',
              'RMSprop(lr=0.01)']

for opt_name in optimizers:

    K.clear_session()
    
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt_name),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
```
* we make the single datafram
```
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([optimizers, metrics_reported],
                                 names=['optimizers', 'metric'])
historydf.columns = idx
```
* we plot
```
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()
```
* the best is Adam

### Lecture 76 - Initialization Code along

* we ll explore the effect of weight initialization with Keras
* when we train our NN we need to assign init values to the weights in our model.
* in our layer creator we can pass a kernel_initializer of our choise and a bias_initializer of our choice. otherwise keras assigns default values 'gloro_uniform and 'zeros' respectively
* in this example we ran again a set of training sessions with various keras kernel(weight) initializers and plot the training results over epochs to compare
```
dflist = []

initializers = ['zeros', 'uniform', 'normal',
                'he_normal', 'lecun_uniform']

for init in initializers:

    K.clear_session()

    model = Sequential()
    model.add(Dense(1, input_shape=(4,),
                    kernel_initializer=init,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    h = model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
    
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
```
* we build the df
```
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([initializers, metrics_reported],
                                 names=['initializers', 'metric'])

historydf.columns = idx
```
* we plot
```
ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title("Accuracy")
plt.xlabel("Epochs")

plt.tight_layout()
```
* lecun_uniform performs best
* everytime we rerun we get a different best... so when we have small datasets initialization has big impact on resutls

### Lecture 77 - Inner Layers Visualization Code Along

* we ll see how to visualize the activations of the inner layers
* its a powerful tool to inspect the model and to dimanson the dataset
* we still deal with banknote dataset (4 numarical feats , 1 categorical target)
* we build a deep network (2 layers deep)
* we train for 20epochs storing the train output 
* we get the evaluation results
```
K.clear_session()

model = Sequential()
model.add(Dense(2, input_shape=(4,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

h = model.fit(X_train, y_train, batch_size=16, epochs=20,
              verbose=1, validation_split=0.3)
result = model.evaluate(X_test, y_test)
```
* we print the result, model.summary() and the model.layers
* model layers are keras Dense objects `<keras.layers.core.Dense at 0x7ff6502acf60>,
 <keras.layers.core.Dense at 0x7ff648764128>]`
 * we can assign a models layer inputs and outputs to vars
 ```
 inp = model.layers[0].input
out = model.layers[0].output
 ```
* they are tensorflow tesors `inp` => *<tf.Tensor 'dense_1_input:0' shape=(?, 4) dtype=float32>* `out` => *<tf.Tensor 'dense_1/Relu:0' shape=(?, 2) dtype=float32>*
* we define a feautre function using Keras backend function method passing in the tensors
```
features_function = K.function([inp], [out])
```
* we checkj it `features_function` => *<keras.backend.tensorflow_backend.Function at 0x7ff641541400>*
* we can now apply this function to any data, we apply it to X_test and it outputs a nested array  where we care for first element [0]. the shape of the data is `features_function([X_test])[0].shape` (412,2) so equal to the input datapoints as 2 equal to the inner layer nodes
* we store it to a var and plot it with hue equal to the labels
```
features = features_function([X_test])[0]
plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')
```
* what we see in the scattterplot is how our model internally learn to separate the labels
* we reset and build a deep er model (3 layers)
```
K.clear_session()

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])
```
* we take the input of 1st layer and output of second and we loop thtough the epochs 1-26 to see the evolution of separation of classes
```
inp = model.layers[0].input
out = model.layers[1].output
features_function = K.function([inp], [out])

plt.figure(figsize=(15,10))

for i in range(1, 26):
    plt.subplot(5, 5, i)
    h = model.fit(X_train, y_train, batch_size=16, epochs=1, verbose=0)
    test_accuracy = model.evaluate(X_test, y_test)[1]
    features = features_function([X_test])[0]
    plt.scatter(features[:, 0], features[:, 1], c=y_test, cmap='coolwarm')
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 4.0)
    plt.title('Epoch: {}, Test Acc: {:3.1f} %'.format(i, test_accuracy * 100.0))

plt.tight_layout()
```

### Lecture 82 - Exercise 3

* KERAS Functional API allows more flexibility than sequential 

### Lecture 84 - Exercise 4

* keras offers the possibility to call a function at each epoch. These are Callbacks. they allow to add functionality

### Lecture 86 - Tensorboard

* we open a terminal in our env an type `tensorboard --logdir='./CourseRepo/course/tmp/udemy'` passing teh dir to the saved files from callbacks
* it launches a web server
* we open 0.0.0.0:6006 in browser
* we have our training metrics in nice diagrams
* a graph of our model

## Section 6 - Convolutional Neural Networks

### Lecture 87 - Features from Pixels

* images and how to use them as input to Neural Networks
* How a computer see an image ? as a grid of points. each point has a value (a byte for grayscale, binary for BW, 3 bytes for RGB). the points on the grid are called pixels
* greyscale usyaly use unsigned int of 1 byte 0-255
* MNIST is a dataset of 70000 hand writen digits. 
* the problem is how to map the image to a class (digit). the image is a 2d table
* the simpest way is to use each pixel as a separate feature. the image is 28x28 so the feature space becomes of size 784. each feature is an integer of 0-255. this approach unrolls the image to a very large array of feats (vector)

### Lecture 88 - MNIST Classification

* we will see how to feed images to a Fully Connected NN with python
* in this approach we have 764 inputs and 10 outputs (classes) passing through a fully connected DNN

### Lecture 89 - MNIST Classification Code Along

* keras offers MNIST in its collection of datasets. we import it `from keras.datasets import mnist`
* mnist offers already split datasets `(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')`
* we check the shape of data `X_train.shape` an is aarray of (60000, 28, 28). test has 10000 samples
* we plot the 1st sample `plt.imshow(X_train[0], cmap='gray')` and see the image
* to use the samples in our Network we need to unroll them with reshape so from 3d we go to 2d array
```
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
```
* also from ints we convert the greyscale val to a 0-1 float
```
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
```
* we ll do hot encoding of labels (0-9 labels) using keras 
```
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
```
* we ll build a fully connected model to classify the images
* we build a 5 layer sequantial model of dense layers
```
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

K.clear_session()

model = Sequential()
model.add(Dense(512, input_dim=28*28, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

```
* we train it and store the results from plotting `h = model.fit(X_train, y_train_cat, batch_size=128, epochs=10, verbose=1, validation_split=0.3)`
* we plot the training jhistory
```
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
```
* in our training we reserved a percentage foro validation. our training accuracy is higher than our validation accuracy so we are overfitting
* we evaluate on our test set and get the accuracy
```
test_accuracy = model.evaluate(X_test, y_test_cat)[1]
test_accuracy
```
* it is 0.98

### Lecture 90 - Beyond Pixels

* we ll look into image feature extraction and local patterns for image recognition
* our previous fully connected model performed well on training but troubled in generalizing
* we llok into better approach (Feature extraction)
* the most simple case of feature extraction is going from an image to a vector of pizels (WE DIT THAT)
* Other methods are:
	* Fourier coefficients
	* Wavelets
	* Histogram of Oriented Gradients (HOG)
	* Speeded Up Robust Features (SURF)
	* Local Binary Patterns (LBP)
	* Color histograms
* The banknote example we use in ML is an example as all feats used in Classification were generated with Feature Extraction on the Image of the BankNote
* These methods require deep domain knowledge to solve a specific problem in image recognition
* With NNs we seek to avoid these methods and learn the best feats from the image itself
* What makes an image different from a vector is that the values of the pixels are correlated in 2d (horizontaly and verticaly). its the 2d pattern that carries the info
* we would like to have a technique to capture these patterns automatically
* we are more interested in a neighborhood of pixels that the position of pixels in the overall image (local patterns)
* the technique is called convolution. to diegest it we need to learn about tensors

### Lecture 91 - Images As Tensors

* Scalars => numbers (no dimensions)
* Vectors => lists of numbers (with dimensions) e.g [4,5,6,7,8,9] 6D vector. numbers in list are coordinates in space
* Matrices => tables of numbers arranged in columns and rows [[0,1,0],[5,0,2]] => 2x3 matrix
* the image is a 2D matrix each entry in matrix is the pixel value (also a list of vectors of same length, for the image a vector can be a row or collumn)
* a vector is a list of scalars
* we can construct an object (tensor) where an object of a certain order can be thought as a list of objects of previous order Order: 0=Scalar, 1=Vector, 2=matrix
* Tensors can be expressed as nested lists of objects of previous order all with the same size
* an order 3 tensor can be thought as a rubik cube a list of 3 tensors of order 2 (matrizes) of size 3x3
* in a tensor we can find numbers by following any of the axes. in a 3d tensor each number is identified by HwD
* The shape of a tensor tells us how many objects there are if we count on a particular axis e.g 3: noshape , [3,3,4,5]: (4,) [[1,2,3],[4,5,6]]: (2,3) [[[0,1],[2,3]],[[4,5],[6,7]]]: (2,2,2)
* a color image is like 3 grayscale images each  corresponding to a color channel (R<G<B). each one contains the pixels of the image in that color. this is an order 3 tensor. there are 2 ordering conventions:
	* If we think the image as a list of 3 single color images (C,H,W)
	* if we think the image as an order 2 list of vector pixels where each pixels is a size 3 vector (R,G,B) then we represent it as (H,W,C) (channel last representation)

### Lecture 92 - Tensor Math Code Along

* we ll see how numpy deals with tensors 
* we create 2 tensors of random ints 0-9 an order 4 of shape (2,3,4,5) and an order 2 size (2,3)
```
A = np.random.randint(10, size=(2, 3, 4, 5))
B = np.random.randint(10, size=(2, 3))
```
* we print them out `B` => *array([[5, 4, 2],[6, 3, 2]])* they are nested numpy.ndarrays
* we can access a single element by passing the indexes along each axis `A[0, 1, 0, 3]` => 8
* we can create a random color image with numpy (3colors, 4x4) `img = np.random.randint(255, size=(4, 4, 3), dtype='uint8')` as h,w,c) channel last
* we plot it out also its channels
```
plt.figure(figsize=(5, 5))
plt.subplot(221)
plt.imshow(img)
plt.title("All Channels combined")

plt.subplot(222)
plt.imshow(img[:, : , 0], cmap='Reds')
plt.title("Red channel")

plt.subplot(223)
plt.imshow(img[:, : , 1], cmap='Greens')
plt.title("Green channel")

plt.subplot(224)
plt.imshow(img[:, : , 2], cmap='Blues')
plt.title("Blue channel")
```
* tensors are numpy arrays so we can do elementwise operations (shape stays same)
```
2 * A
A+A
```
* we can do dot multiplication with .tensordot() which is an extension of the dot product and it allows us to specify which axes we are contracting on  `np.tensordot(A, B, axes=([0, 1], [0, 1]))` shape is (4,5) and result is *array([[ 79,  86,  62,  85, 111],
       [ 63, 132,  49, 136, 116],
       [ 93, 116,  77,  86, 118],
       [110, 103, 111,  84, 124]])*
 * so essentialy we lose the axes we contract on (2,3)
 * if we contract on one `np.tensordot(A, B, axes=([0], [0])).shape` we are left with (3,4,5,3) so 2 is lost from both and 3 from b is added in the end as dimansion

### Lecture 93 - Convolution in 1D

* say we have 2 arrays g [-1,1] and f [0,0,0,0,0,1,1,1,1,1]
* the convolution of the 2 is defined as follows:
	* we take a subarray from f equal in size to the array g (smaller) starting from index 0. we multiply each eleemnt in g with the equivalene indexed element of subarray f and add the products (-1x0+1x0 = 0) we place the result in the first position of a new array [0, ....] this is step n=0
	* we shift the window inf by 1 position to the right n=1 and repear the process. the result array is nmow [0,0,...]
	* we continue till we reach the end of array f . the result is f * g =  [0,0,0,0,1,0,0,0,0]
	* the mathematic formula oof 1d convolution is `(f*g)[n]=Σm=-MtoM(f[n-m]g[m]`
* convolution takes max value when the f value matches value g. we will use convolution to detect patterns in images

### Lecture 94 - Convolution in 1D Code Along

* we create 2 arrays a and b
```
a = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype='float32')
b = np.array([-1, 1], dtype='float32')
```
* we use numpy built in method .convolve() to get the convolution between the 2 whitch is *array([ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0., 0.,  0.,  0.], dtype=float32)*
* we plot the original array and the convoluted
```
plt.subplot(211)
plt.plot(a, 'o-')

plt.subplot(212)
plt.plot(c, 'o-')
```

### Lecture 95 - Convolution in 2D

* say we have a 9by9 image of black background (pixel value=-1) and a white x (pixel value 1)
* we will look for diagonal patterns like the ones rerpesented in a tegel of size 3by3 [[1,-1,-1],[-1,1,-1],[-1,-1,1]]
* in the image there are 3 subsections that match exactly the patterns, and some other places with similar pattern
* we calculate the convolution of the image with the tegel by multiplying the 3by3 tegel with a moving 3by3 subsection (step of 1)of the image and storing the results
* when we have a perfect match the result of matrix multplication of the tegel and the subsection is [[1,1,1],[1,1,1],[1,1,1]] and the sum is 9 which we divide by the area of the patch 9/9=1. we store the result in a new image at the pixel location that represents the center pixel of the moving subsections current position
* we repeat the process for the whole image and produce a heatmap for the result convoluted image values. we can see immediatly the perfect matches convolution is represented with a cross in a circle symbol

### Lecture 96 - Image Filters Code Along

* we will perform convolutions on images using scipy library
* we import our convolutiuon methods and a utils lib
```
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from scipy import misc
```
* we import an image from the miscellaneous lib `img = misc.ascent()`
* we check its size `img.shape` its (512,512) so we expect it to be grayscale
* we plot it `plt.imshow(img, cmap='gray')`
* we ll do a convolution of this image with a kernel 3by3 expressed as numpy array
```
h_kernel = np.array([[ 1,  2,  1],
                     [ 0,  0,  0],
                     [-1, -2, -1]])
```
* we view the kernel `plt.imshow(h_kernel, cmap='gray')` we expect to get the highlights of horizontal lines
* we do the convolution using convolve2d() and show the result
```
res = convolve2d(img, h_kernel)
plt.imshow(res, cmap='gray')
```
* in a neural network the kernel is found by the network and is not predefined as its values are weights

### Lecture 97 - Convolutional Layers

* convolution layers, convolution of tensors, strides and padding
* if we convolve the same image with different filters we get different convolved images, each of these convolved images represents the locations of the matches with the corresponding filter
* we can take many filters and arrange them in a convolutional layer. when we feen an image to this layer the output is a stack of convolved images
* since we know about tensors and we know that all the convolved images have same size (the filters have same size) we can arrange them in a tensor where the number of channels corresponds to the number of filters
* filters also can be arranges in a tensor (a tensor of weights in the convolutional layer)
* Our *Input Tensor* is an order 4 tensor (N,H,W,C)
	* N: number of images (in batch)
	* H: height of image 
	* W: width of image
	* C: Number of color channels
* In MNIST assuming we feed all im,ages to the network in a single batch (CRAZY!) the input tensor is (60000,28.28,1)
* In a similar way we can stack the filters in a Convolutional Layer Tensor which will also be an order 4 tensor (Hf,Wf,Ci,Co)
	* Hf: Height of filter patch
	* Wf: Width of filter patch
	* Ci: Channels in Input (color channels)
	* Co: Channels in output (#filters)
* so if we use 2 3x3 filters for mnist set the COnv tensor size will be (3,3,1,2)
* if we use these filters to convolve a single channel 9by9 image the size will be 
	* INPUT: (-,9,9,1)
	* CONV: (3,3,1,2)
	* RESULT: (-,7,7,2) 7 is H and W of convolved image
* As RESULT is an order 4 tensor we can feed it again to a filter layer
	* INPUT: (-,7,7,2)
	* CONV: (3,3,2,2) (4 fitlers 2 per input, same)
	* RESULT: (-,5,52)
* The convolved images are smaller after each convolution. this is controlled (apart from the size of filter) by the stride or pixel step we take in both directions. in all our previous examples stride was (1,1)
* for our previous example a (2,2) stride would result in a 5,5 image and a step of (3,3) to a 3,3
* we can stride on different length on each direction stride of (3,1) would result to a (3,7) image
* if we dont want to loose the borders as essentialy even for a 3x3 filter we loose the outer rim in the result we need to add padding (with 0s) nd get as result a 9x9 image. this convolution preserves the size and is called Same

### Lecture 98 - Convolutional Layers Code Along

* we will use a convolution layer from Keras to perform a filtering operation on an image
* we import the layer type `from keras.layers import Conv2D`
* we use the same image from previous codealong
* we build a order 4 input tensor reshaping the image array `img_tensor = img.reashape(1,512,512,1)`
* we build a simple NN (sequential) with 1layer (Conv2D) and compil it (adam optimizer, mse metric)
```
model = Sequential()
model.add(Conv2D(1,(3,3),strides=(2,1),input_shape=(512,512,1)))
model.compile('adam','mse')
```
* the params for the conv2D
	* 1 : node count
	* (3,3) : filter size
	* strides=(2,1) strides
	* input_shape=(512,512,1) HxWxC
* we dont train the model but it still can perform a predict= feed forward pass (single convolution)
```
img_pred_tensor = model.predict(img_tensor)
```
* the result has shape (1,255,510,1) stride of 3 halves the result, no padding so we lose the rim
* we extract the image from tensor with indexing and plot it
```
img_pred = img_pred_tensor[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')
```
* image is compressed verically
* we check the kernel used (shape and plot)
```
weights = model.get_weights()
weights[0].shape # (3,3,1,1)
plt.imshow(weights[0][:, :, 0, 0], cmap='gray')
weights[0] = np.ones(weights[0].shape)
```
* we custom set an all ones filter and set it to model and use it to predict (do the convolution)
```
weights[0] = np.ones(weights[0].shape)
model.set_weights(weights)
img_pred_tensor = model.predict(img_tensor)
```
* we plot the rewsult
```
img_pred = img_pred_tensor[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')
```
* we get a blurred image
* in the conv2D we can use padding='same' to add a size 1 padding of 0

### Lecture 99 - Pooling Layers

* a pooling operation reduces the size of the image by discarding some information
* maxpooling only preserves (keeps) the maximum value in a patch and stores it in the new image by discarding the values of the other pixels
* usually pooling patches do not overlap (the size of the image is reduced)
* if we apply 2x2 max pooling to all the convoluyted images of the previous Convolution Layers Lecture their size is reduced from 7x7 to 4x4. the reduced images still preserve the approximate location of a perfect match
* a pooling layer operates on the width and height axis of the data tensor (usually of the convolution layer output) (-,7,7,3) => (2,2) pooling layer => (-,4,4,3)
* an AVG (Average) Pooling layer does the sam like maxpooling but instead of keeping the max pixel calculates and keeps the average
* POOLING LAYER reduces size of image preserving the image information

### Lecture 100 - Pooling Layers Code Along

* we import the  pooling layers from keras `from keras.layers import MaxPool2D, AvgPool2D`
* we build a simple model like before with one pooling layer (max pooling 5x5)
* if we dont define strides it equalts to pool size
```
model = Sequential()
model.add(MaxPool2D((5, 5), input_shape=(512, 512, 1)))
model.compile('adam', 'mse')
```
* we dont train but use the model to get the result. we directly extract the image from tensor with indexing for plotting 
```
img_pred = model.predict(img_tensor)[0, :, :, 0]
plt.imshow(img_pred, cmap='gray')
```
* granularity appers (max) info is preserved
* we do the same with av gpooling `AvgPool2D` and compare the result image (smooth result)

### Lecture 101 - Convolutional Neural Networks

* we combine all what we learned to build a CNN
* we start by stacking conv and pool layers together
* we start from a simple image to smaller sized&more channel images
* when we finish we can use the output of stacked layers as features of a fully connected classification network
* feature extraction stack uses COnvolution-RELU-Convolution-RELU-Pooling-Convolution-RELU-Pooling 
* the result is multiple 2x2 features. 
* we can unroll the feats to a long vector and connect this vector to labels using a fully connected layer (or multiple fully connected layers)
* Image->COnvolution-RELU-Convolution-RELU-Pooling-Convolution-RELU-Pooling-FullyConnected-FullyConnected-> Labels
* the deeper we go in the network we get richer and more unique patterns matched so we get more robust classification

### Lecture 102 - Convolutional Neural Networks Code Along

* we will use the MNIST dataset input.
* we reshape our X data (train,test) into tensors (-1 means leave as is)
```
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train.shape # (60000,28,28,1)
```
* we import more layer types `from keras.layers import Flatten, Activation`
* we clear the session `K.clear_session()`
* set a sequential model `model = Sequential()`
* we build the feature extraction part
	* conv layer of 32 filters (3,3)
	* maxpool of (2,2)
	* relu activation layer
```
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Activation('relu'))
```
* we flatten the output tensor to a long array
```
model.add(Flatten())
```
* we add 2 dense layers for classification (multiclass so softmax in 2nd 10 nodes like the classes )
```
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```
* we compile the model
```
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```
* we check the model config with `model.summary()`
* the initial number of params for our classification is 692352
* we fit the model `model.fit(X_train, y_train_cat, batch_size=128, epochs=2, verbose=1, validation_split=0.3)`
* and evaluate it with test set `model.evaluate(X_test, y_test_cat)` we get 97.5% accuracy

### Lecture 103 - Weights in CNNs

* the filters in CNNs are not predefined. the va;lues in their tensors are the weights the network has to learn to solve the classification problem. so when we run a CNN training session we force it to learn the best patterns to solve the problem
* We ll compare the parameter count in a fully connected layer and a convolutional layer
	* Input data: Image 10x10 => 100 nums
	* Fully Connected layer: 32 nodes => 100 weights each => 3200 weights
	* Convolutional Layer: 32 filters or nodes 3x3 => 9 weights each => 288 weights
* also as our arrays in a Conv lyaer are set in a 2D shape is more probalbe to catch local info in both directions
* In CNN we trade params for computation

### Lecture 104 - Beyond Images

* other uses of CNNs  and Limits CNNs
* CNNs are used on sound using the spectrogram (rerpesent sound in 2 axes, x: time steps, y: intensity in each frequency band)
* we can feed spectrogram to CNN and treat it like an image
* we can map a sentence of text on an image. (vertical axis: words in dictionary, vertical axis: position in sentence)
* CNNs are NOT useful in some cases:
	* they are good at capturing spatialpatterns data
	* not good if local patterns dont exist (eg in a DB table, order plays no role)

### Section 7 - Cloud GPUs

### Lecture 107 - Floyd GPU notebook setup

* sign up floyd (free plan)
* install or update floyd cli (in conda env) `Install or update Floyd `
* login to cli `floyd login`
* initialize a project `floyd init zerotodeeplearning`
* make it public
* go to notebook and addd `floyd run --gpu` in front of constly command (train cifar10)

## Section 8 - Recurrent Neural Networks

### Lecture 108 - Time Series

* Time series examples
	* stock market
	* music
	* video games
	* IoT
	* Energy monitoring
	* Earthquakes
	* weather
	* Biosensors
	* Website monitoring analytics
* Time series: ordered series of datapoints
* Time series can have as values, simple numbers or vectors
* time series of vectors is represented with a time series per vector component
* vector sequence time series: car speed, stero sound (2 channels)
* ML for Time Series
	* Prediction of future (regression, forecasting)
	* Pattern Recognition & segmentation (classification,clustering, anomaly detection)
	* Compression, noise reduction (preprocessing)
* Forecasting:  given a sequence of values we want to predict the future values in the sequence (regression type problem but non linear ergression)
* Detect Anomalies: given a sequence with regular behaviour identify when it deviates from the regularity (supervised learning approach if we know the anomalies we are looking for, or unsupervised learning if we dont know in that case we predict future values and we compare actual future values with the predicted ones)
* Classification: find recurring patterns in time series and identify regions of recurring patterns (supervised, unsupervised)
* Our data is ordered in time so caution is required
	* we should avoid leaking future information in the future used by the model
	* we cannot do a random split in train and test values, we need to split in time where training data are happening before the split and test data after the split
	* sometimes a trnad is clearly visible in a time series (data related with human activity) we should remove these repetitions or consider the periodicity as a feat

### Lecture 109 - Sequence Problems

* other types of problems involving time sequences
* simplest machine learnign problem involving a time sequence? an 1to1 problem
	* 1 data input (data point or tensor) to the model
	* model generates a prediction using the input
* all problems we hve come acroos fall in this category: linear regression, classification, image classification with CNN are 1to1 problems. e.g in digits classification (MNIST) for each input image we had one label, when we classify purchases from users for each user we have a purchase label, when we distinguish banknotes. for each banknote we have one label
* In time series we extend this formula for the model to use the past values of the input and the output
* One to many problems starts like the 1to1 problem, we have 1 input to the model and it generates 1 output but the output of the model is fed back to the model as a new input and network genreates a new output and so on. 
* a typical example of 1tomany problem is image captioning: as ingle image generates as output a text description of n length
* Many to one problem is the reverse: we feed multiple input to the network, and at each step we feed back the network output into the network. we do this untill we reachthe end of the input sequence where we get the final output (text sentiment analysis falls in this category)
* Many to many: in that case we have a sequence in input and a sequence in output (with feedback) this is the case of text translation
* Synchronous Many to many: network outputs a value at each input (using the input and its previous state) e.g video frame classification
* Recurrent neural Networks can deal with all these problems as the node interconnections form a directed cycle.
* RNNs are able to retain state between iterations using their network output in the next step (like infinite response filters (IIR) in DSP) its like running a fixed program with fixed input and internal vars
* RNNs essentialy describe programs (THey are TURNING complete)
	* Vanilla NNs : optimiztion over functions
	* RNNs optimization over programs

### Lecture 110 - Vanilla RNN

* Vanilla RNN = Simpest Form of RNN, make it Deep
* RNNs maintain an internal state by using their own output as part of the input for th enext prediction
* A simpe RNN can be viewed as a Fully Connected NN if we unroll it in the time axis
	* output at time t ht=tanh(wht-1+uxt) we use 2 weights u and w (looks like EWMA) 
	* w,u do not depend on time (same weights at all times)
* we can build Deep RNNs by stacking Recurrent units (nodes) one on top of the other. we feed the input to first layer and feed the output of first layer to second ht@2 = tanh(w@2ht-1@2 + u@2ht@1) and so on
* THe simple RNN works ony for short term memory (problematic on long term due to vanishing gradients)

### Lecture 111 - LSTM and GRU

 * LSTM (Long Short Term Memory Cell)
 * GRU (Gated Recurrent Unit Cell)
 * Both dont suffer from Vanishing Gradients problem
 * The Vanilaa RNN where the output is fed back as input suffers from Vanishing Gradients Problem and cannot capture long term dependencies in sequences
 * A brilliant way to solve thsi problem is LSTM
 	* this network is organized in cells
 	* cell maintains state
 	* gates modify information
 	* first gate is called "FORGET" gate `ft = σ(Wf[ht-1,xt]+bf` a sigmoid layer that takes the output@t-1 and current input x@t concatenates them in a single tensor applies a linear transformation (wx+b) followed by a sigmoid activation. because of sigmoid the output of this gate is 0-1. this output multiplies the cell state @t-1 and therefore is called the 'FORGET' gate
 	* the second gate is called the "INPUT" gate `it=σ(Wi[ht-1,xt]+bi)` it takes the previous output@t-1 and new input@t and passes them into another sigmoid layer like the forget gate. like before it returns a value 0-1
 	* the value of th einput gate is multiplied with the output of a *Candidate* layer C~t = tanh(Wc[ht-1,xt]+bC) which returns a candidate vector to be oadded to the  internal state
 	* The new cell state is calcualted Ct=ftCt-1+itC~t
 	* The 'Output' gate control how much of the internal state is passed in the output `ot = σ(Wo[ht-1,xt]+bo)` => `ht = ottanh(Ct)`
 * A simpler version of LSTM is GRU
 	* similar to LSTM but simplified
 	* it doesnt pass along 2 separate variables (state, output) only the output
 	* has 2 gates instead of 3: 1st gate controls the mixing of the previous output with current input, the mix is passd to a tanh gate for the output 2nd gate controls mixing of previous output with the current output
 * GRU formuals
 	* zt=δ(Wz[ht-1,xt])
 	* rt = s(Wr[ht-1,xt])
 	* h~t = tanh(W[rt * ht-1,xt])
 	* ht = (1-zt) * ht-1 + zt * h~t (EWMA)

 ### Lecture 112 - Time Series Forecasting Code Along

 * we import the usual libs
 ```
 import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
 ```

 * we import our data from csv, skip some data and show the df
 ```
 df = pd.read_csv('../data/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9,
                 engine='python')
df.head()
 ```
 * the data is from canada bureau of statistcs and has the monthly nationwide sales from 1991 to preasent (Jan-2017) it contains 3 columns (date,unadjusted sales, seasinly adjusted sales)
 * we want ot have date as index we use a panda timeseries lib `from pandas.tseries.offsets import MonthEnd`
 * we pass the column to .todatetime() method using MonthEnd() (which gives the month endo for each month) and set it as index
 ```
 df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
df.head()
 ```
 * we plot the df `df.plot()` we can clearly see the 2008 financial crisis as an anomaly in the otherwise periodical series with an upward trend
 * as we said we have to split our data on a specific time not randomly (past=train,future=test)
 * we set a split date `split_date = pd.Timestamp('01-01-2011')`
 * we use the unadjusted data for our dataset spliting on split date
 ```
train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]
 ```
 * we plot them
 ```
 ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
 ```
 * we will use minmaxscaler to scale our data for ML. when we use scalers its critical to fit only on our train data (we must know nothing for the future)
 ```
 from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)
 ```
 * from our dataset we produce our inputs and outputs.
 * in timeseries inputs and outputs come from the same timeseries data but outputs start from the next timestamp (1step shift usually) as our model learn from previous timestamp and tries to match the next
 ```
 X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]
 ```
 * we start our effort with a Vanilla NN (no recrsion)
 * we do our imports
 ```
 from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
 ```
 * it is a regression problem (we try to predict a numeric val)
 * we build our model (Fully COnnected NN)
 ```
 K.clear_session()

model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
 ```
 * we instantiate our earlystopping callback (sve CPU )
 ```
 early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
 ```
 * we train our model
 ```
 model.fit(X_train, y_train, epochs=200,
          batch_size=2, verbose=1,
          callbacks=[early_stop])
 ```
 * we get our predictions from the model `y_pred = model.predict(X_test)`
 * we counterplot predictions and truth to evaluate 
 ```
 plt.plot(y_test)
plt.plot(y_pred)
```
* our model is not good. it imitated the periodicity of the timeseries but it is time shifted by one step and has an offset

### Lecture 113 - Time Series Forecasting with LSTM Code Along

* we import LSTM layer from KEras `from keras.layers import LSTM`
* recurrent layer requires an input with shape (batch_size, timesteps, input_dim) for our case (batch_size,1,1)
* our tran input has shape `X_train.shape` (239,1)
* we will transhorm it to X_train[:, None] with shape (239, 1, 1)
```
X_train_t = X_train[:, None]
X_test_t = X_test[:, None]
```
* we build our model (1 layer, 6 cells) still a regression problem
```
K.clear_session()
model = Sequential()

model.add(LSTM(6, input_shape=(1, 1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
```
* we train our model  `model.fit(X_train_t, y_train,epochs=100, batch_size=1, verbose=1,callbacks=[early_stop])`
* we evaluate and compare results like before
```
y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
```
* it recalls fetter than the ANN but is not so much better

### Lecture 114 - Rolling Windows

* a technique to extract feats from a time series.
* in all the proplems so far we asked from NNs to predict the future using their internal state and the current value of the sequence
* what id instead of feeding just one step we fed a short history of values
* this is called the rolling window approach. we take a fixed size window and use all the data contained in the window to predict values after the window. then we slide the windo by a stride (usually 1/2 window length)and repeat
* The rolling windows approiach is used in many successful apps (it was used much before NNs)
* It can be used with ML and traditional feats
* Features at window:
	* Past Raw Values
	* Stats: Mean<Stdev,Moments,Correlation ...
	* Frequency: FFT< Wavelets, Spectral Band Power, Entropy...
	* Autoregressive models
	* Other: Hjorth,Mann-Kendall...
* In deep learning on windows we have 2 options on how to use the window:
	* feed the window in a RNN in sequence 1 point at a time asking for a prediction at the end (Many to 1 sequence problem)
	* feed the window all at once using each point as a different feature, if we do this we are not bount to use RNNs. we can use CNNs or FNNs because we have mapped the regional sequence problem to an 1to1 problem where input space is a vector (points in window)

### Lecture 115 - Rolling Windows Code Along

* we will use an RNN that uses a window and also a fnn that uses a window
* we build a scaled time series (from our scaled data which is a numpy array build a dataframe)
```
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
```
* in this scaled dfs we add time shifted columns from 1 to 13 steps shift using the pandas .shift() method
```
for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
```