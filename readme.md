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