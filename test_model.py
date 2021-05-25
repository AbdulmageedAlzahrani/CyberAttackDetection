import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
def nDataRows_X(dataSet, cols):
    return np.loadtxt('dataset/data' + np.str(dataSet) + '.csv', delimiter=",", skiprows=1, usecols=cols)

def nDataRows_Y(dataSet, cols):
    return np.loadtxt('dataset/data' + np.str(dataSet) + '.csv', delimiter=",", skiprows=1, usecols=cols, dtype=np.str)

def getFeaturesTitle(dataSet, features):
    return np.loadtxt('dataset/data' + np.str(dataSet) + '.csv', delimiter=",", skiprows=0, max_rows = 1, usecols=features, dtype=np.str)

#features = np.array([29, 51, 41, 31, 107, 14, 33, 109, 91, 87, 115, 49, 99, 121, 89, 22, 60, 16, 84, 80, 85, 20, 113, 70])
features = np.arange(128)
dataSetNumber = 1
# print(getFeaturesTitle(dataSetNumber, features))

X = nDataRows_X(dataSetNumber, features)
Y = nDataRows_Y(dataSetNumber, 128)

Y[Y == 'Attack'] = 1
Y[Y == 'Natural'] = 0

#local_path = r"C:/Users/moh_2/Desktop/192/ICS 481/ANN_Project/‏‏data" + np.str(dataSetNumber) + r"_files/"
local_path = ''

X = np.nan_to_num(X)  #solving the ValueError: Input contains infinity or a value too large for dtype('float64').
Scaler = joblib.load(local_path + 'scaler_data' + np.str(dataSetNumber) + '.gz')
X = Scaler.fit_transform(X)

history = joblib.load(local_path + 'history_data' + np.str(dataSetNumber) + '.gz')
train = joblib.load(local_path + 'trainSet_data' + np.str(dataSetNumber) + '.gz')
test =  joblib.load(local_path + 'testSet_data' + np.str(dataSetNumber) + '.gz')
bestModel = load_model(local_path + 'model_dataSet' + np.str(dataSetNumber) + '.h5')
bestModel.summary()

loss, accuracy = bestModel.evaluate(X[test],Y[test], batch_size=512,verbose=1)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

epochs = np.arange(len(history.history['accuracy']))

plt.plot(epochs, history.history['accuracy'])
plt.title('accuracy vs epoch for: dataSet' + np.str(dataSetNumber))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(epochs, history.history['loss'])
plt.title('loss vs epoch for: dataSet' + np.str(dataSetNumber))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# dataSet    epochs      batch_size      CFS     Loss      accuracy   
#   1         1000           512         NO      0.04        94.76%   
#   2         1000           512         NO      0.05        95.27%   
#   3         1000           512         NO      0.04        95.57%   
#   4         1000           512         NO      0.07        91.73%   
#   5         1000           512         NO      0.05        94.57%   
#   6         1000           512         NO      0.07        92.94%   
#   7         1000           512         NO      0.04        94.65%   
#   8         1000           512         NO      0.06        93.41%   
#   9         1000           512         NO      0.09        90.45%   
#   10        1000           512         NO      0.08        90.83%   
#   11        1000           512         NO      0.07        92.95%   
#   12        1000           512         NO      0.06        93.68%   
#   13        1000           512         NO      0.07        93.18%   
#   14        1000           512         NO      0.05        94.73%   
#   15        1000           512         NO      0.04        94.89%    