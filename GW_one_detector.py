from __future__ import print_function
from matplotlib import pyplot as plt
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


import os
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['GOTO_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['openmp'] = 'True'

#os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile

obj = SampleFile()
obj.read_hdf("default_diff_1.hdf")
df = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj1 = SampleFile()
#obj1.read_hdf("default_snr10_40000.hdf")
#df1 = obj1.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj2 = SampleFile()
#obj2.read_hdf("default_snr-10_60000.hdf")
#df2 = obj2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj3 = SampleFile()
#obj3.read_hdf("default_2048_sector10.hdf")
#df3 = obj3.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj4 = SampleFile()
#obj4.read_hdf("default_2048_sector44.hdf")
#df4 = obj4.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj5 = SampleFile()
#obj5.read_hdf("default_2048_sector45.hdf")
#df5 = obj5.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj2 = SampleFile()
#obj2.read_hdf("default_diff_1.hdf")
#df2 = obj2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#df = pd.concat([df1, df2], ignore_index= True)

obj_test = SampleFile()
obj_test.read_hdf("default_test_set.hdf")
df_test = obj_test.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file



# Same labels will be reused throughout the program
#LABELS = ['1',
#          '2',
#          '3',
#          '4',
#          '5',
#          '6',
#          '7',
#          '8']
# The number of steps within one time segment
#TIME_PERIODS = 512
TIME_PERIODS = 512
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
#STEP_DISTANCE = 256
STEP_DISTANCE = 512

#Extracting seconds_before_event from 'info.csv' file

#dataset = pd.read_csv('info.csv')
#sbe = dataset.iloc[:,1].values


#Creating Timestamp column
timestamp = []
event_time = 1234567936
count = 0
grid = []
grid_test = []
for i in range(100000):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid.append(np.linspace(event_time - 0.20, event_time + 0.05, int(2048 * 0.25)))
for i in range(100):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid_test.append(np.linspace(event_time - 0.20, event_time + 0.05, int(2048 * 0.25)))
    
timestamp = np.hstack(grid)#timestamp is now the array representing the required Timestamp column of the datastructure
timestamp_test = np.hstack(grid_test)

#new = df[['h1_signal', 'l1_signal', 'v1_signal']].copy() # Extracting h1_signal, l1_signal and v1_signal columns
                                                         # from hdf file.
    
new = df[['h1_strain']].copy() 
new_test = df_test[['h1_strain']].copy()

h1 = new.iloc[:,0] #Extracting h1_signal from 'new' dataframe
#l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe
#v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
h1_test = new_test.iloc[0:100,0]

#Creating h1_signal, l1_signal and v1_signal columns 
h1_signal_col = np.hstack(h1)
#l1_signal_col = np.hstack(l1)
#v1_signal_col = np.hstack(v1) #hstack is used for creating a list of lists into a numpy array.
h1_signal_col_test = np.hstack(h1_test)

#Taking Fourier transform of the signals
#h1_signal_col = np.fft.fft(np.hstack(h1))
#l1_signal_col = np.fft.fft(np.hstack(l1))
#v1_signal_col = np.fft.fft(np.hstack(v1)) #hstack is used for creating a list of lists into a numpy array.


#Extracting ra and dec angles from the hdf file
#angles = df[['ra', 'dec']].copy()

# For training set:
angles = df[['ra', 'dec']].copy()
#ra = angles.iloc[:,0].values
#dec = angles.iloc[:,1].values

angles['ra'] = angles['ra'].astype(np.float64)
angles['dec'] = angles['dec'].astype(np.float64)

ra = 2.0*np.pi*angles['ra']
dec = np.arcsin(1.0 - 2.0*angles['dec'])

# For test set:
angles_test = df_test[['ra', 'dec']].copy()
#ra = angles.iloc[:,0].values
#dec = angles.iloc[:,1].values

angles_test['ra'] = angles_test['ra'].astype(np.float64)
angles_test['dec'] = angles_test['dec'].astype(np.float64)

ra_test = 2.0*np.pi*angles_test['ra']
dec_test = np.arcsin(1.0 - 2.0*angles_test['dec'])

    
np.savetxt('RA_test.txt', np.transpose(ra_test))
np.savetxt('Dec_test.txt', np.transpose(dec_test))
#################################################### For 4096 sectors #################################################################
#a = -np.pi/32.0
#b = (np.pi/2.0 + np.pi/64.0)
#ra_array = []
#dec_array = []
#for i in range(65):
#    ra_array.append(a+np.pi/32.0)
#    a = a+np.pi/32.0
#for j in range(65):
#    dec_array.append(b-np.pi/64.0)
#    b = b-np.pi/64.0
    
#k = 0
#multi_list = [[0 for i in range(64)] for j in range(64)]
#for j in range(64):
#    for i in range(64):
#        k = k + 1
#        multi_list[j][i]= k 

#print(multi_list)

#label = []
#for i in range(100000):
#    for j in range(64):
#        if(ra[i] >= ra_array[j] and ra[i] <= ra_array[j+1]):
#            ra_index = j
#            break
#    for k in range(64):
#        if(np.sin(dec[i]) >= np.sin(dec_array[k+1]) and np.sin(dec[i]) <= np.sin(dec_array[k])):
#            dec_index = k
#            break
#    label.append(multi_list[dec_index][ra_index])

# For test set:   

#a_test = -np.pi/32.0
#b_test = (np.pi/2.0 + np.pi/64.0)
#ra_array_test = []
#dec_array_test = []
#for i in range(65):
#    ra_array_test.append(a_test+np.pi/32.0)
#    a_test = a_test+np.pi/32.0
#for j in range(65):
#    dec_array_test.append(b_test-np.pi/64.0)
#    b_test = b_test-np.pi/64.0
    
#k_test = 0
#multi_list_test = [[0 for i in range(64)] for j in range(64)]
#for j in range(64):
#    for i in range(64):
#        k_test = k_test + 1
#        multi_list_test[j][i]= k_test 

#print(multi_list_test)

#label_test = []
#for i in range(4000):
#    for j in range(64):
#        if(ra_test[i] >= ra_array_test[j] and ra_test[i] <= ra_array_test[j+1]):
#            ra_index_test = j
#            break
#    for k in range(64):
#        if(np.sin(dec_test[i]) >= np.sin(dec_array_test[k+1]) and np.sin(dec_test[i]) <= np.sin(dec_array_test[k])):
#            dec_index_test = k
#            break
#    label_test.append(multi_list_test[dec_index_test][ra_index_test])

##################################################### For 2048 sectors ################################################################

#a = -np.pi/32.0
#b = (np.pi/2.0 + np.pi/32.0)
#ra_array = []
#dec_array = []
#for i in range(65):
#    ra_array.append(a+np.pi/32.0)
#    a = a+np.pi/32.0
#for j in range(33):
#    dec_array.append(b-np.pi/32.0)
#    b = b-np.pi/32.0
    
    
#k = 0
#multi_list = [[0 for i in range(64)] for j in range(32)]
#for j in range(32):
#    for i in range(64):
#        k = k + 1
##        multi_list[j][i]= str(k)
#        multi_list[j][i]= k

#print(multi_list)
    
#label = []
#for i in range(100004):
#    for j in range(64):
#        if(ra[i] >= ra_array[j] and ra[i] <= ra_array[j+1]):
#            ra_index = j
#            break
#    for k in range(32):
#        if(dec[i] >= dec_array[k+1] and dec[i] <= dec_array[k]):
#            dec_index = k
#            break
#    label.append(multi_list[dec_index][ra_index])

    
    
# For test set:

#a_test = -np.pi/32.0
#b_test = (np.pi/2.0 + np.pi/32.0)
#ra_array_test = []
#dec_array_test = []
#for i in range(65):
#    ra_array_test.append(a_test+np.pi/32.0)
#    a_test = a_test+np.pi/32.0
#for j in range(33):
#    dec_array_test.append(b_test-np.pi/32.0)
#    b_test = b_test-np.pi/32.0
    
    
#k_test = 0
#multi_list_test = [[0 for i in range(64)] for j in range(32)]
#for j in range(32):
#    for i in range(64):
#        k_test = k_test + 1
##        multi_list[j][i]= str(k)
#        multi_list_test[j][i]= k_test

#print(multi_list)
    
#label_test = []
#for i in range(4000):
#    for j in range(64):
#        if(ra_test[i] >= ra_array_test[j] and ra_test[i] <= ra_array_test[j+1]):
#            ra_index_test = j
#            break
#    for k in range(32):
#        if(dec_test[i] >= dec_array_test[k+1] and dec_test[i] <= dec_array_test[k]):
#            dec_index_test = k
#            break
#    label_test.append(multi_list_test[dec_index_test][ra_index_test])
#    label_test.append(multi_list_test[dec_index_test][ra_index_test])
#    label_test.append(multi_list_test[dec_index_test][ra_index_test])


##################################################### For 8192 sectors ################################################################

a = -np.pi/64.0
b = (np.pi/2.0 + np.pi/64.0)
ra_array = []
dec_array = []
for i in range(129):
    ra_array.append(a+np.pi/64.0)
    a = a+np.pi/64.0
for j in range(65):
    dec_array.append(b-np.pi/64.0)
    b = b-np.pi/64.0  

k = 0
multi_list = [[0 for i in range(128)] for j in range(64)]
for j in range(64):
    for i in range(128):
        k = k + 1
        multi_list[j][i]= k 

print(multi_list)

label = []
for i in range(100000):
    for j in range(128):
        if(ra[i] >= ra_array[j] and ra[i] <= ra_array[j+1]):
            ra_index = j
            break
    for k in range(64):
        if(dec[i] >= dec_array[k+1] and dec[i] <= dec_array[k]):
            dec_index = k
            break
    label.append(multi_list[dec_index][ra_index])

# For test set:

a_test = -np.pi/64.0
b_test = (np.pi/2.0 + np.pi/64.0)
ra_array_test = []
dec_array_test = []
for i in range(129):
    ra_array_test.append(a_test+np.pi/64.0)
    a_test = a_test+np.pi/64.0
for j in range(65):
    dec_array_test.append(b_test-np.pi/64.0)
    b_test = b_test-np.pi/64.0  

k_test = 0
multi_list_test = [[0 for i in range(128)] for j in range(64)]
for j in range(64):
    for i in range(128):
        k_test = k_test + 1
        multi_list_test[j][i]= k_test 

print(multi_list_test)

label_test = []
for i in range(100):
    for j in range(128):
        if(ra_test[i] >= ra_array_test[j] and ra_test[i] <= ra_array_test[j+1]):
            ra_index_test = j
            break
    for k in range(64):
        if(dec_test[i] >= dec_array_test[k+1] and dec_test[i] <= dec_array_test[k]):
            dec_index_test = k
            break
    label_test.append(multi_list_test[dec_index_test][ra_index_test])

    
    
    
#For making the required dataset, the label should be mentioned for each row of the total input dataset.
sector = []
for i in range(100000):
    a = label[i]
    for j in range(512):
        sector.append(a)
#sector now is the array representing the label of the sample at each row of the input dataset.

sector_test = []
for i in range(100):
    a_test = label_test[i]
    for j in range(512):
        sector_test.append(a_test)
        
        
#Creating the actual final dataframe from the individual arrays as columns

d = {"H1_Signal": h1_signal_col, "Sector": sector, "Timestamp": timestamp}

d_test = {"H1_Signal": h1_signal_col_test,"Sector": sector_test, "Timestamp": timestamp_test}

# d is the dictionary of the input dataframe.

#Converting d to a pandas dataframe
data = pd.DataFrame(d)
data_test = pd.DataFrame(d_test)

def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

# Normalize features for training data set
#data_train["H1_Signal"] = feature_normalize(data["H1_Signal"])
data["H1_Signal"] = feature_normalize(data["H1_Signal"])
data_test["H1_Signal"] = feature_normalize(data_test["H1_Signal"])
#data_train["L1_Signal"] = feature_normalize(data["L1_Signal"])
#data_train["V1_Signal"] = feature_normalize(data["V1_Signal"])
# Round in order to comply to NSNumber from iOS
#data_train = data_train.round({"H1_Signal": 6, "L1_Signal": 6, "V1_Signal": 6})
#data_train = data_train.round({"H1_Signal": 6})
data = data.round({"H1_Signal": 6})
data_test = data_test.round({"H1_Signal": 6})

LABEL = "Sector"
def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 1
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df["H1_Signal"].values[i: i + time_steps]
#        ys = data["L1_Signal"].values[i: i + time_steps]
#        zs = data["V1_Signal"].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
#        segments.append([xs, ys, zs])
        segments.append([xs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

#x_train, y_train = create_segments_and_labels(data_train,
#                                              TIME_PERIODS,
#                                              STEP_DISTANCE,
#                                              LABEL)

x_train, y_train = create_segments_and_labels(data,
                                  TIME_PERIODS,
                                  STEP_DISTANCE,
                                  LABEL)

x_test, y_test = create_segments_and_labels(data_test,
                                  TIME_PERIODS,
                                  STEP_DISTANCE,
                                  LABEL)

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
print('x_test shape: ', x_test.shape)
# Displays (20869, 40, 3)
#print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train shape: ', y_train.shape)
print('y_test shape:', y_test.shape)
# Displays (20869,)

# Set input & output dimensions
#num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_time_periods, num_sensors = 512, 1
#num_classes = le.classes_.size
#print(list(le.classes_))

num_time_periods_test, num_sensors_test = 512, 1
#num_classes_test = le_test.classes_.size
#print(list(le_test.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (num_time_periods*num_sensors)
input_shape_test = (num_time_periods_test*num_sensors_test)

x_train = x_train.reshape(x_train.shape[0], input_shape)
x_test = x_test.reshape(x_test.shape[0], input_shape)

print('x_train shape:', x_train.shape)
# x_train shape: (20869, 120)
print('input_shape:', input_shape)
# input_shape: (120)

print('x_test shape:', x_test.shape)
# x_train shape: (20869, 120)
print('input_shape:', input_shape_test)
# input_shape: (120)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")


# %%

# One-hot encoding of y_train labels (only execute once!)
#y_train = np_utils.to_categorical(y_train, num_classes)
#print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%

onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded),
a = y_train.reshape(-1,1)
onehot_encoded = onehot_encoder.fit_transform(a)
#print(onehot_encoded)

#onehot_encoder_test = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded),
a_test = y_test.reshape(-1,1)
onehot_encoded_test = onehot_encoder.transform(a_test)
#print(onehot_encoded)


print("\n--- Create neural network model ---\n")

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 50, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 50, activation='relu'))
model_m.add(MaxPooling1D(3))
#model_m.add(Conv1D(100, 55, activation='relu'))
#model_m.add(Conv1D(100, 55, activation='relu'))
#model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 50, activation='relu'))
model_m.add(Conv1D(160, 50, activation='relu'))
model_m.add(GlobalAveragePooling1D())
#model_m.add(Dropout(0.5))
#model_m.add(Dense(num_classes, activation='softmax'))
model_m.add(Dense(7934, activation='softmax'))
print(model_m.summary())
# Accuracy on training data: 99%
# Accuracy on test data: 91%

# %%


print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 6000
EPOCHS = 200

# Hyper-parameters
BATCH_SIZE = 6000
EPOCHS = 200

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
#history = model_m.fit(x_train,
#                      onehot_encoded,
#                      batch_size=BATCH_SIZE,
#                      epochs=EPOCHS,
#                      callbacks=callbacks_list,
#                      validation_split=0.2,
#                      verbose=1)

# %%

history = model_m.fit(x_train, onehot_encoded, batch_size = 6000, epochs = 200, verbose=1)


#print("\nAccuracy on test data: %0.2f" % score[1])
#print("\nLoss on test data: %0.2f" % score[0])

#y_pred_test = classifier.predict(x_test)

y_pred_test = model_m.predict(x_test)


# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(onehot_encoded_test, axis=1)

np.savetxt('Scores_1.out', np.transpose([max_y_pred_test,max_y_test]))


for i in range(100):
    np.savetxt("Prediction_files/Preds_"+str(i)+".txt", np.transpose(y_pred_test[i]))
    

score = model_m.evaluate(x_test, onehot_encoded_test, verbose=1)
print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])
