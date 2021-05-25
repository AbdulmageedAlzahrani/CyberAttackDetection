import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based.CFS import cfs
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
import joblib

def nDataRows_X(dataSet, cols):
    return np.loadtxt('dataset/data' + np.str(dataSet) + '.csv', delimiter=",", skiprows=1, usecols=cols)

def nDataRows_Y(dataSet, cols):
    return np.loadtxt('dataset/data' + np.str(dataSet) + '.csv', delimiter=",", skiprows=1, usecols=cols, dtype=np.str)

def getFeaturesTitle(dataSet, features):
    return np.loadtxt('dataset/data' + np.str(dataSet) + '.csv', delimiter=",", skiprows=0, max_rows = 1, usecols=features, dtype=np.str)

#features = np.array([29, 51, 41, 31, 107, 14, 33, 109, 91, 87, 115, 49, 99, 121, 89, 22, 60, 16, 84, 80, 85, 20, 113, 70])
features = np.arange(128)

#finished 2, 3, 4, 5, 6
dataSetNumber = 4

# print(getFeaturesTitle(dataSetNumber, features))

X = nDataRows_X(dataSetNumber, features)
Y = nDataRows_Y(dataSetNumber, 128)

Y[Y == 'Attack'] = 1
Y[Y == 'Natural'] = 0

X = np.nan_to_num(X)  #solving the ValueError: Input contains infinity or a value too large for dtype('float64').
Scaler = MinMaxScaler()
X = Scaler.fit_transform(X)
  
print(cfs(X,Y))
exit(0)

kf = StratifiedKFold(10, shuffle=True , random_state=100)
fstart, fend, fold = 0, 18, 0
acc = np.zeros(10)
train_index, test_index = -1, -1
# print(kf.split(X,Y))

#local_path = r"C:/Users/moh_2/Desktop/192/ICS 481/ANN_Project/‏‏data" + np.str(dataSetNumber) + r"_files/"
local_path = ''
for train , test in kf.split(X,Y):
    model = Sequential([
        Dense(80,input_shape=(128,),bias_initializer="random_uniform",activation="sigmoid"),
        Dense(50,bias_initializer="random_uniform",activation="sigmoid"),
        Dense(40,bias_initializer="random_uniform",activation="sigmoid"),
        Dense(30,bias_initializer="random_uniform",activation="sigmoid"),
        Dense(1,bias_initializer="random_uniform",activation="sigmoid")
    ])

    # print('train: ' , train)
    model.compile(Adam(lr=0.01),loss="mse",metrics=["accuracy"])

    history = model.fit(X[train], Y[train], epochs=1000, batch_size=512, verbose=1)
    #NOTE: in each epoch you will find in the output that number of training data is less than the total rows of the dataset, that because K-Fold seperate the training set from the test set
    # print(history)
    loss, accuracy = model.evaluate(X[test],Y[test], batch_size=512,verbose=1)
    # print('result: ', result)    
    acc[fold] = accuracy
    if acc[fold] == np.amax(acc):
        train_index = train
        test_index = test
        print("\nThe bestModel history so far: \n            Loss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
        joblib.dump(history, local_path + 'history_data' + np.str(dataSetNumber) + '.gz')
        joblib.dump(train, local_path + 'trainSet_data' + np.str(dataSetNumber) + '.gz')
        joblib.dump(test, local_path +  'testSet_data' + np.str(dataSetNumber) + '.gz')
        model.save(local_path + 'model_dataSet' + np.str(dataSetNumber) + '.h5')
    fold = fold + 1
joblib.dump(Scaler, local_path + 'scaler_data' + np.str(dataSetNumber) + '.gz')

model.summery()

