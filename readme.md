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
* R@ is good and similar