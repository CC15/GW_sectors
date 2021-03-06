from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
#import scikitplot as skplt

#%pylab inline


import os
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['GOTO_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['openmp'] = 'True'

import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile


obj1 = SampleFile()
obj1.read_hdf("default_snr.hdf")
df1 = obj1.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj1 = SampleFile()
#obj1.read_hdf("default_diff.hdf")
#df1 = obj1.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#obj2 = SampleFile()
#obj2.read_hdf("default_diff_1.hdf")
#df2 = obj2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#obj_test = SampleFile()
#obj_test.read_hdf("default_test_set.hdf")
#df_test = obj_test.as_dataframe(True,True,True,False)

obj2 = SampleFile()
obj2.read_hdf("default_2048_sector9.hdf")
df2 = obj2.as_dataframe(True,True,True,False)

obj3 = SampleFile()
obj3.read_hdf("default_2048_sector10.hdf")
df3 = obj3.as_dataframe(True,True,True,False)

obj4 = SampleFile()
obj4.read_hdf("default_2048_sector44.hdf")
df4 = obj4.as_dataframe(True,True,True,False)

obj5 = SampleFile()
obj5.read_hdf("default_2048_sector45.hdf")
df5 = obj5.as_dataframe(True,True,True,False)

#obj_test5 = SampleFile()
#obj_test5.read_hdf("default_G333462.hdf")
#df_test5 = obj_test5.as_dataframe(True,True,True,False)

#obj_test6 = SampleFile()
#obj_test6.read_hdf("default_G330308.hdf")
#df_test6 = obj_test6.as_dataframe(True,True,True,False)

#obj_test7 = SampleFile()
#obj_test7.read_hdf("default_event1_snr10.hdf")
#df_test7 = obj_test7.as_dataframe(True,True,True,False)

#obj_test8 = SampleFile()
#obj_test8.read_hdf("default_event1_snr15.hdf")
#df_test8 = obj_test8.as_dataframe(True,True,True,False)

#obj_test9 = SampleFile()
#obj_test9.read_hdf("default_event1_snr20.hdf")
#df_test9 = obj_test9.as_dataframe(True,True,True,False)

#obj_test10 = SampleFile()
#obj_test10.read_hdf("default_2048_sector45.hdf")
#df_test10 = obj_test10.as_dataframe(True,True,True,False)

#obj_l1 = SampleFile()
#obj_l1.read_hdf("default_event3_snr45.hdf")
#df_l1 = obj_l1.as_dataframe(True,True,True,False)

#obj_v1 = SampleFile()
#obj_v1.read_hdf("default_event1_snr45.hdf")
#df_v1 = obj_v1.as_dataframe(True,True,True,False)

obj_test = SampleFile()
obj_test.read_hdf("default_test_P-P_plot.hdf")
df_test = obj_test.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

df = pd.concat([df1, df2, df3, df4, df5, df_test], ignore_index= True)

data = df[['h1_strain', 'l1_strain', 'v1_strain', 'event_time', 'injection_snr']].copy()

#data_l1 = df_l1[['h1_strain', 'l1_strain', 'v1_strain', 'event_time', 'injection_snr']].copy()

#df_test = obj_test.as_dataframe(True,True,True,False)
#df_test2 = pd.concat([df_test,df_test1], ignore_index= True)

# Extracting the strain values.

# For training set: 
from scipy.signal import hilbert, chirp
import numpy as np
h1 = data.iloc[:,0]
l1 = data.iloc[:,1]
#v1 = data.iloc[:,2]

# Creating array of ratios of sum of ten highest ampltudes for each sample after taking Hilbert transforms:

# For training set:
h1_sum = 0.0
l1_sum = 0.0
#v1_sum = 0.0
h1_l1_sum = []
#l1_v1_sum = []
#h1_v1_sum = []

for i in range(102604):
    h1_sum = 0.0
    l1_sum = 0.0
#    v1_sum = 0.0
    h1_des = np.abs(hilbert(h1[i]))
    l1_des = np.abs(hilbert(l1[i]))
#    v1_des = np.abs(hilbert(v1[i]))
    h1_des.sort()
    l1_des.sort()
#    v1_des.sort()
    for j in range(10):
        h1_sum = h1_sum + h1_des[511 - j]
        l1_sum = l1_sum + l1_des[511 - j]
#        v1_sum = v1_sum + v1_des[511 - j]
        
    h1_l1_sum.append(h1_sum/l1_sum)
#    l1_v1_sum.append(l1_sum/v1_sum)
#    h1_v1_sum.append(h1_sum/v1_sum)
    
# Creating the time array

timestamp = []
event_time = 1234567936
count = 0
for i in range(512):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid = np.linspace(event_time - 0.20, event_time + 0.05, int(2048 * 0.25))
    
# Extracting the angles from the dataframe

# For training set:
angles = df[['ra', 'dec']].copy()
#ra = angles.iloc[:,0].values
#dec = angles.iloc[:,1].values

angles['ra'] = angles['ra'].astype(np.float64)
angles['dec'] = angles['dec'].astype(np.float64)

ra = 2.0*np.pi*angles['ra']
dec = np.arcsin(1.0 - 2.0*angles['dec'])

a = -np.pi/32.0
b = (np.pi/2.0 + np.pi/32.0)
ra_array = []
dec_array = []
for i in range(65):
    ra_array.append(a+np.pi/32.0)
    a = a+np.pi/32.0
for j in range(33):
    dec_array.append(b-np.pi/32.0)
    b = b-np.pi/32.0
    
    
k = 0
multi_list = [[0 for i in range(64)] for j in range(32)]
for j in range(32):
    for i in range(64):
        k = k + 1
##        multi_list[j][i]= str(k)
        multi_list[j][i]= k

print(multi_list)
    
label = []
for i in range(102604):
    for j in range(64):
        if(ra[i] >= ra_array[j] and ra[i] <= ra_array[j+1]):
            ra_index = j
            break
    for k in range(32):
        if(np.sin(dec[i]) >= np.sin(dec_array[k+1]) and np.sin(dec[i]) <= np.sin(dec_array[k])):
            dec_index = k
            break
    label.append(multi_list[dec_index][ra_index])

# Time delays of original signals from cross-correlations.            
            
            
# For training set:            
from scipy.signal import correlate

h1l1_time = []
#l1v1_time = []
#h1v1_time = []

h1l1_time_h = []
#l1v1_time_h = []
#h1v1_time_h = []

N = 512

time = np.arange(1-N,N)

for i in range(102604):
    h1l1_time.append(time[correlate(h1[i],l1[i]).argmax()])
#    l1v1_time.append(time[correlate(l1[i],v1[i]).argmax()])
#    h1v1_time.append(time[correlate(h1[i],v1[i]).argmax()])
    
    
# Time delays of Hilbert-transformed signals from cross-correlations.

# For training set:
for i in range(102604):
    h1_h = hilbert(h1[i])
    l1_h = hilbert(l1[i])
#    v1_h = hilbert(v1[i])
    h1l1_time_h.append(time[correlate(np.abs(h1_h),np.abs(l1_h)).argmax()])
#    l1v1_time_h.append(time[correlate(np.abs(l1_h),np.abs(v1_h)).argmax()])
#    h1v1_time_h.append(time[correlate(np.abs(h1_h),np.abs(v1_h)).argmax()])
    
    
# Maximum value of cross-correlations of original signals.    

# For training set:
h1l1_sum = []
#l1v1_sum = []
#h1v1_sum = []


for i in range(102604):
    h1l1_sum.append(correlate(h1[i],l1[i]).max())
#    l1v1_sum.append(correlate(l1[i],v1[i]).max())
#    h1v1_sum.append(correlate(h1[i],v1[i]).max())
    
    
# Average phase lags of original signals around merger

# For training set:
h1l1_phase = []
#l1v1_phase = []
#h1v1_phase = []

h1l1_freq = []
#l1v1_freq = []
#h1v1_freq = []

h1_10 = []
l1_10 = []
#v1_10 = []

fs = 2048.0

for i in range(102604):
    h1_10 = []
    l1_10 = []
#    v1_10 = []
    h1_des = hilbert(h1[i])
    l1_des = hilbert(l1[i])
#    v1_des = hilbert(v1[i])
    h1_des.sort()
    l1_des.sort()
#    v1_des.sort()
    for j in range(10):
        h1_10.append(h1_des[511-j])
        l1_10.append(l1_des[511-j])
#        v1_10.append(v1_des[511-j])


    h1l1_phase.append(np.average((np.unwrap(np.angle(h1_10))) - (np.unwrap(np.angle(l1_10)))))
#    l1v1_phase.append(np.average((np.unwrap(np.angle(l1_10))) - (np.unwrap(np.angle(v1_10)))))
#    h1v1_phase.append(np.average((np.unwrap(np.angle(h1_10))) - (np.unwrap(np.angle(v1_10)))))
    

# Angle between the vectors representing the original signals.

# For training set:
h1l1_angle = []
#l1v1_angle = []
#h1v1_angle = []
    
for i in range(102604):
    h1_h = hilbert(h1[i])
    l1_h = hilbert(l1[i])
#    v1_h = hilbert(v1[i])
    c_h1l1 = np.inner(h1_h, np.conj(l1_h))/ np.sqrt(np.inner(h1_h, np.conj(h1_h)) * np.inner(l1_h, np.conj(l1_h)))
#    c_l1v1 = np.inner(l1_h, np.conj(v1_h))/ np.sqrt(np.inner(l1_h, np.conj(l1_h)) * np.inner(v1_h, np.conj(v1_h)))
#    c_h1v1 = np.inner(h1_h, np.conj(v1_h))/ np.sqrt(np.inner(h1_h, np.conj(h1_h)) * np.inner(v1_h, np.conj(v1_h)))
    h1l1_angle.append(np.angle(c_h1l1))
#    l1v1_angle.append(np.angle(c_l1v1))
#    h1v1_angle.append(np.angle(c_h1v1))
    

# Maximum value of cross-correlations of Hilbert-transformed signals.

# For training set:
h1l1_sum_h = []
#l1v1_sum_h = []
#h1v1_sum_h = []
        
for i in range(102604):
    h1_h = hilbert(h1[i])
    l1_h = hilbert(l1[i])
#    v1_h = hilbert(v1[i])
    h1l1_sum_h.append(correlate(np.abs(h1_h),np.abs(l1_h)).max())
#    l1v1_sum_h.append(correlate(np.abs(l1_h),np.abs(v1_h)).max())
#    h1v1_sum_h.append(correlate(np.abs(h1_h),np.abs(v1_h)).max())
    
    
# Creating the pandas dataframe with the input features and the labels for the neural network

# For training set:
#df_1 = {"H1/L1": h1_l1_sum, "L1/V1": l1_v1_sum, "H1/V1": h1_v1_sum, "H1-L1 Time lag": h1l1_time, "L1-V1 Time lag": l1v1_time, "H1-V1 Time lag": h1v1_time, "H1-L1 Phase lag": h1l1_phase, "L1-V1 Phase lag": l1v1_phase, "H1-V1 Phase lag": h1v1_phase, "H1/L1 Corr": h1l1_sum, "L1/V1 Corr": l1v1_sum, "H1/V1 Corr": h1v1_sum, "H1-L1 Angle": h1l1_angle, "L1-V1 Angle": l1v1_angle, "H1-V1 Angle": h1v1_angle, "H1/L1 Corr Hilbert": h1l1_sum_h, "L1/V1 Corr Hilbert": l1v1_sum_h, "H1/V1 Corr Hilbert": h1v1_sum_h, "H1-L1 Time Hilbert": h1l1_time_h, "L1-V1 Time Hilbert": l1v1_time_h, "H1-V1 Time Hilbert": h1v1_time_h, "Sector": label}

df_1 = {"H1/L1": h1_l1_sum, "H1-L1 Time lag": h1l1_time, "H1-L1 Phase lag": h1l1_phase, "H1/L1 Corr": h1l1_sum,"H1-L1 Angle": h1l1_angle, "H1/L1 Corr Hilbert": h1l1_sum_h, "H1-L1 Time Hilbert": h1l1_time_h, "Sector": label}


dataframe = pd.DataFrame(df_1)


# Define column name of the label vector
#LABEL = 'SectorEncoded'
# Transform the labels from String to Integer via LabelEncoder
#le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
#dataframe[LABEL] = le.fit_transform(dataframe['Sector'].values.ravel())

#dataframe_test[LABEL] = le.fit_transform(dataframe_test['Sector'].values.ravel())


# Splitting into input features and labels

X_train = dataframe.iloc[:, 0:7].values
#y_train = dataframe.iloc[:, 22].values
y_train = dataframe.iloc[:,7].values
y_train = y_train.reshape(len(y_train), 1)


# Set input & output dimensions
#num_time_periods = X_train.shape[1]
#num_classes = le.classes_.size
#print(list(le.classes_))

# Before continuing, we need to convert all feature data (x_train) and label data (y_train) into a datatype accepted by Keras.
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)


#One last step we need to do is to conduct one-hot-encoding of our labels. Please only execute this line once!
#y_train_hot = np_utils.to_categorical(y_train, num_classes)
#print('New y_train shape: ', y_train_hot.shape)

onehot_encoder = OneHotEncoder(sparse=False)
##integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(y_train)
#print(onehot_encoded)

##y_train_hot_new = []
##for i in range(199999):
##    y_train_hot_new.append(onehot_encoded[i])

y_test_hot = []
for i in range(100004,102604):
    y_test_hot.append(onehot_encoded[i])

y_train_hot_new = []
for i in range(100004):
    y_train_hot_new.append(onehot_encoded[i])
    
X_train_new = []
for i in range(100004):
    X_train_new.append(X_train[i])

X_test = []
for i in range(100004,102604):
    X_test.append(X_train[i])
    
    
# Making the ANN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', input_dim = 7))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the third layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fourth layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fifth layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fifth layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(p = 0.2))

# Adding the fifth layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))

# Adding the fifth layer
#classifier.add(Dense(units = 200, kernel_initializer = 'uniform'))
#classifier.add(BatchNormalization())
#classifier.add(Activation('relu'))
#classifier.add(Dropout(p = 0.2))



# Adding the output layer
classifier.add(Dense(units = 2048, kernel_initializer = 'uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('softmax'))
print(classifier.summary())


#X_train_new = []
#for i in range(199999):
#    X_train_new.append(X_train[i])

a = np.asarray(X_train_new)
b = np.asarray(y_train_hot_new)
c = np.asarray(X_test)
d = np.asarray(y_test_hot)

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set

#history = classifier.fit(X_train, y_train_hot, batch_size = 2000, epochs = 300, callbacks = callbacks_list, validation_split = 0.02, verbose=1)

history = classifier.fit(a, b, batch_size = 2000, epochs = 300, verbose=1)


training_acc = np.asarray(history.history['acc'])
training_loss = np.asarray(history.history['loss'])
print('\nAccuracy on training data: %0.2f' % training_acc[-1])
print('\nLoss on training data: %0.2f' % training_loss[-1])

score = classifier.evaluate(c, d, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])


y_pred_test = classifier.predict(c)

# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test_hot, axis=1)


np.savetxt('Scores_1.out', np.transpose([max_y_pred_test,max_y_test]))



for i in range(2600):
    np.savetxt("Prediction_files/Preds_"+str(i)+".txt", np.transpose(y_pred_test[i]))
    

#classifier.save("model.h5")
#print("Saved model to disk")

   



