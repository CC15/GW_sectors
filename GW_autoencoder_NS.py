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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D,Input, LSTM, RepeatVector
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras.utils import np_utils

import numpy as np
import pandas as pd
from SampleFileTools1 import SampleFile

obj = SampleFile()
obj.read_hdf("default_NS_snr-50.hdf")
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
obj_test.read_hdf("default_NS_test.hdf")
df_test = obj_test.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test2 = SampleFile()
#obj_test2.read_hdf("default_GW170817_freq_500.hdf")
#df_test2 = obj_test2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file


#df_test = pd.concat([df_test1, df_test2], ignore_index= True)

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
for i in range(25):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid_test.append(np.linspace(event_time - 0.20, event_time + 0.05, int(2048 * 0.25)))
    
timestamp = np.hstack(grid)#timestamp is now the array representing the required Timestamp column of the datastructure
timestamp_test = np.hstack(grid_test)

def normalize(a):
    new_array = []
    for i in range(100000):
        dataset = a[i]
        maximum = np.max(dataset)
        minimum = np.abs(np.min(dataset))
        for j in range(512):
            if(dataset[j] > 0):
                dataset[j] = dataset[j]/maximum
            else:
                dataset[j] = dataset[j]/minimum
        new_array.append(dataset)
    return new_array
        
def normalize_test(a):
    new_array = []
    for i in range(25):
        dataset = a[i]
        maximum = np.max(dataset)
        minimum = np.abs(np.min(dataset))
        for j in range(512):
            if(dataset[j] > 0):
                dataset[j] = dataset[j]/maximum
            else:
                dataset[j] = dataset[j]/minimum
        new_array.append(dataset)
    return new_array
        
#new = df[['h1_signal', 'l1_signal', 'v1_signal']].copy() # Extracting h1_signal, l1_signal and v1_signal columns
                                                         # from hdf file.
    
new = df[['l1_strain']].copy() 
new_test = df_test[['l1_strain']].copy()

new_pure = df[['l1_signal']].copy() 
new_test_pure = df_test[['l1_signal']].copy()


l1 = new.iloc[:,0] #Extracting h1_signal from 'new' dataframe
#l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe

#v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
l1_test = new_test.iloc[0:25,0]

l1_pure = new_pure.iloc[:,0] #Extracting h1_signal from 'new' dataframe
#l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe
#v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
l1_test_pure = new_test_pure.iloc[0:25,0]

l1_new = normalize(l1)
l1_test_new = normalize_test(l1_test)

l1_pure_new = normalize(l1_pure)
l1_test_pure_new = normalize_test(l1_test_pure)
    
#for i in range(25):
#    h1_test = normalize(h1_test[i])
#    h1_test_pure = normalize(h1_test_pure[i])


#Creating h1_signal, l1_signal and v1_signal columns 
l1_signal_col = np.hstack(l1_new)
#l1_signal_col = np.hstack(l1)
#v1_signal_col = np.hstack(v1) #hstack is used for creating a list of lists into a numpy array.
l1_signal_col_test = np.hstack(l1_test_new)

#Creating h1_signal, l1_signal and v1_signal columns 
l1_signal_col_pure = np.hstack(l1_pure_new)
#l1_signal_col = np.hstack(l1)
#v1_signal_col = np.hstack(v1) #hstack is used for creating a list of lists into a numpy array.
l1_signal_col_test_pure = np.hstack(l1_test_pure_new)

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
for i in range(100000):
    for j in range(64):
        if(ra[i] >= ra_array[j] and ra[i] <= ra_array[j+1]):
            ra_index = j
            break
    for k in range(32):
        if(dec[i] >= dec_array[k+1] and dec[i] <= dec_array[k]):
            dec_index = k
            break
    label.append(multi_list[dec_index][ra_index])

    
    
# For test set:

a_test = -np.pi/32.0
b_test = (np.pi/2.0 + np.pi/32.0)
ra_array_test = []
dec_array_test = []
for i in range(65):
    ra_array_test.append(a_test+np.pi/32.0)
    a_test = a_test+np.pi/32.0
for j in range(33):
    dec_array_test.append(b_test-np.pi/32.0)
    b_test = b_test-np.pi/32.0
    
    
k_test = 0
multi_list_test = [[0 for i in range(64)] for j in range(32)]
for j in range(32):
    for i in range(64):
        k_test = k_test + 1
##        multi_list[j][i]= str(k)
        multi_list_test[j][i]= k_test

#print(multi_list)
    
label_test = []
for i in range(25):
    for j in range(64):
        if(ra_test[i] >= ra_array_test[j] and ra_test[i] <= ra_array_test[j+1]):
            ra_index_test = j
            break
    for k in range(32):
        if(dec_test[i] >= dec_array_test[k+1] and dec_test[i] <= dec_array_test[k]):
            dec_index_test = k
            break
    label_test.append(multi_list_test[dec_index_test][ra_index_test])
#    label_test.append(multi_list_test[dec_index_test][ra_index_test])
#    label_test.append(multi_list_test[dec_index_test][ra_index_test])

#For making the required dataset, the label should be mentioned for each row of the total input dataset.
sector = []
for i in range(100000):
    a = label[i]
    for j in range(512):
        sector.append(a)
#sector now is the array representing the label of the sample at each row of the input dataset.

sector_test = []
for i in range(25):
    a_test = label_test[i]
    for j in range(512):
        sector_test.append(a_test)
    
#Creating the actual final dataframe from the individual arrays as columns

d = {"L1_Signal": l1_signal_col, "Sector": sector, "Timestamp": timestamp}

d_test = {"L1_Signal": l1_signal_col_test, "Sector": sector_test, "Timestamp": timestamp_test}

d_pure = {"L1_Signal": l1_signal_col_pure, "Sector": sector, "Timestamp": timestamp}

d_test_pure = {"L1_Signal": l1_signal_col_test_pure, "Sector": sector_test, "Timestamp": timestamp_test}


# d is the dictionary of the input dataframe.

#Converting d to a pandas dataframe
data = pd.DataFrame(d)
data_test = pd.DataFrame(d_test)

data_pure = pd.DataFrame(d_pure)
data_test_pure = pd.DataFrame(d_test_pure)

#def feature_normalize(dataset):

#    mu = np.mean(dataset, axis=0)
#    sigma = np.std(dataset, axis=0)
#    return (dataset - mu)/sigma

# Normalize features for training data set
#data_train["H1_Signal"] = feature_normalize(data["H1_Signal"])
#data["H1_Signal"] = feature_normalize(data["H1_Signal"])
#data_test["H1_Signal"] = feature_normalize(data_test["H1_Signal"])

#data_pure["H1_Signal"] = feature_normalize(data_pure["H1_Signal"])
#data_test_pure["H1_Signal"] = feature_normalize(data_test_pure["H1_Signal"])


data_l1 = data.iloc[:,0] #Extracting h1_signal from 'new' dataframe
#l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe

#v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
#data_h1_test = data_test.iloc[0:25,0]
data_l1_test = data_test.iloc[0:25,0]

data_l1_pure = new_pure.iloc[:,0] #Extracting h1_signal from 'new' dataframe
#l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe
#v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
#h1_test_pure = new_test_pure.iloc[0:25,0]
l1_test_pure = new_test_pure.iloc[0:25,0]




#data_train["L1_Signal"] = feature_normalize(data["L1_Signal"])
#data_train["V1_Signal"] = feature_normalize(data["V1_Signal"])
# Round in order to comply to NSNumber from iOS
#data_train = data_train.round({"H1_Signal": 6, "L1_Signal": 6, "V1_Signal": 6})
#data_train = data_train.round({"H1_Signal": 6})
data = data.round({"L1_Signal": 6})
data_test = data_test.round({"L1_Signal": 6})

data_pure = data_pure.round({"L1_Signal": 6})
data_test_pure = data_test_pure.round({"L1_Signal": 6})


LABEL = "Sector"
def create_segments_and_labels_train_noisy(df_func, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 1
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df_func) - time_steps, step):
        xs = df_func["L1_Signal"].values[i: i + time_steps]
#        ys = data["L1_Signal"].values[i: i + time_steps]
#        zs = data["V1_Signal"].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_func[label_name][i: i + time_steps])[0][0]
#        segments.append([xs, ys, zs])
        segments.append([xs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

LABEL = "Sector"
def create_segments_and_labels_test_noisy(df_func, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 1
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df_func) - time_steps, step):
        xs = df_func["L1_Signal"].values[i: i + time_steps]
#        ys = data["L1_Signal"].values[i: i + time_steps]
#        zs = data["V1_Signal"].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_func[label_name][i: i + time_steps])[0][0]
#        segments.append([xs, ys, zs])
        segments.append([xs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

LABEL = "Sector"
def create_segments_and_labels_train_pure(df_func, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 1
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df_func) - time_steps, step):
        xs = df_func["L1_Signal"].values[i: i + time_steps]
#        ys = data["L1_Signal"].values[i: i + time_steps]
#        zs = data["V1_Signal"].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_func[label_name][i: i + time_steps])[0][0]
#        segments.append([xs, ys, zs])
        segments.append([xs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

LABEL = "Sector"
def create_segments_and_labels_test_pure(df_func, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 1
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df_func) - time_steps, step):
        xs = df_func["L1_Signal"].values[i: i + time_steps]
#        ys = data["L1_Signal"].values[i: i + time_steps]
#        zs = data["V1_Signal"].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_func[label_name][i: i + time_steps])[0][0]
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

x_train_noisy, y_train_noisy = create_segments_and_labels_train_noisy(data,
                                  TIME_PERIODS,
                                  STEP_DISTANCE,
                                  LABEL)

x_test_noisy, y_test_noisy = create_segments_and_labels_test_noisy(data_test,
                                  TIME_PERIODS,
                                  STEP_DISTANCE,
                                  LABEL)

x_train_pure, y_train_pure = create_segments_and_labels_train_pure(data_pure,
                                  TIME_PERIODS,
                                  STEP_DISTANCE,
                                  LABEL)

x_test_pure, y_test_pure = create_segments_and_labels_test_pure(data_test_pure,
                                  TIME_PERIODS,
                                  STEP_DISTANCE,
                                  LABEL)


print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train_noisy shape: ', x_train_noisy.shape)
print('x_test_noisy shape: ', x_test_noisy.shape)
# Displays (20869, 40, 3)
#print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train_noisy shape: ', y_train_noisy.shape)
print('y_test_noisy shape:', y_test_noisy.shape)
# Displays (20869,)

# Inspect x data
print('x_train_pure shape: ', x_train_pure.shape)
print('x_test_pure shape: ', x_test_pure.shape)
# Displays (20869, 40, 3)
#print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train_pure shape: ', y_train_pure.shape)
print('y_test_pure shape:', y_test_pure.shape)
# Displays (20869,)

# Set input & output dimensions

num_time_periods, num_sensors = 512, 1

num_time_periods_test, num_sensors_test = 512, 1

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (num_time_periods*num_sensors)
input_shape_test = (num_time_periods_test*num_sensors_test)

x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], input_shape)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], input_shape)

x_train_pure = x_train_pure.reshape(x_train_pure.shape[0], input_shape)
x_test_pure = x_test_pure.reshape(x_test_pure.shape[0], input_shape)


print('x_train_noisy shape:', x_train_noisy.shape)

print('input_shape:', input_shape)


print('x_test_noisy shape:', x_test_noisy.shape)

print('input_shape:', input_shape_test)


print('x_train_pure shape:', x_train_pure.shape)

print('input_shape:', input_shape)


print('x_test_pure shape:', x_test_pure.shape)

print('input_shape:', input_shape_test)



# Convert type for Keras otherwise Keras cannot process the data
x_train_noisy = x_train_noisy.astype("float32")
y_train_noisy = y_train_noisy.astype("float32")

x_test_noisy = x_test_noisy.astype("float32")
y_test_noisy = y_test_noisy.astype("float32")


# Convert type for Keras otherwise Keras cannot process the data
x_train_pure = x_train_pure.astype("float32")
y_train_pure = y_train_pure.astype("float32")

x_test_pure = x_test_pure.astype("float32")
y_test_pure = y_test_pure.astype("float32")

# %%

# One-hot encoding of y_train labels (only execute once!)
#y_train = np_utils.to_categorical(y_train, num_classes)
#print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%

onehot_encoder_noisy = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded),
a_noisy = y_train_noisy.reshape(-1,1)
onehot_encoded_noisy = onehot_encoder_noisy.fit_transform(a_noisy)
#print(onehot_encoded)

#onehot_encoder_test = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded),
a_test_noisy = y_test_noisy.reshape(-1,1)
onehot_encoded_test_noisy = onehot_encoder_noisy.transform(a_test_noisy)
#print(onehot_encoded)


onehot_encoder_pure = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded),
a_pure = y_train_pure.reshape(-1,1)
onehot_encoded_pure = onehot_encoder_pure.fit_transform(a_pure)
#print(onehot_encoded)

#onehot_encoder_test = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded),
a_test_pure = y_test_pure.reshape(-1,1)
onehot_encoded_test_pure = onehot_encoder_pure.transform(a_test_pure)
#print(onehot_encoded)

#input_sig = Input(batch_shape=(None,512,1))
#x = Conv1D(64,3, activation='relu', padding='valid')(input_sig)
#x1 = MaxPooling1D(2)(x)
#x2 = Conv1D(32,3, activation='relu', padding='valid')(x1)
#x3 = MaxPooling1D(2)(x2)
#flat = Flatten()(x3)
#encoded = Dense(32,activation = 'relu')(flat)

#print("shape of encoded {}".format(K.int_shape(flat)))

#x2_ = Conv1D(32, 3, activation='relu', padding='valid')(x3)
#x1_ = UpSampling1D(2)(x2_)
#x_ = Conv1D(64, 3, activation='relu', padding='valid')(x1_)
#upsamp = UpSampling1D(2)(x_)
#flat = Flatten()(upsamp)
#decoded = Dense(512,activation = 'relu')(flat)
#decoded = Reshape((512,1))(decoded)

#print("shape of decoded {}".format(K.int_shape(x1_)))


#input_sig = Input(batch_shape=(None,512,1))
#x = Conv1D(32,16, padding='same', activation='tanh', name='conv_1')(input_sig)
#x = MaxPooling1D(pool_size=2, padding='same')(x)
#x = Conv1D(16,16, padding='same', activation='tanh', name='conv_2')(x)
#x = MaxPooling1D(pool_size=2, padding='same')(x)
#x = Conv1D(8,16, padding='same', activation='tanh', name='conv_3')(x)
#x = MaxPooling1D(pool_size=2, padding='same')(x)

#x = Conv1D(8,16, padding='same', activation='tanh', name='conv_4')(x)
#x = UpSampling1D(size=2)(x)
#x = Conv1D(8,16, padding='same', activation='tanh', name='conv_5')(x)
#x = UpSampling1D(size=2)(x)
#x = Conv1D(8,16, padding='same', activation='tanh', name='conv_6')(x)
#x = UpSampling1D(size=2)(x)

#output = Conv1D(1,1,strides=1, activation='tanh', padding='same')(x)

model = Sequential()
model.add(Reshape((512,1), input_shape=(input_shape,)))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(16,16, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(8,16, padding='same', activation='tanh'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Flatten())
#model.add(Dense(100))
model.add(RepeatVector(1))
model.add(LSTM(100, activation='tanh', return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(100, activation='tanh', return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(100, activation='tanh', return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(100, activation='tanh', return_sequences=True))
model.add(LSTM(100, activation='tanh'))
#model.add(Dropout(0.2))
#model.add(Flatten())
model.add(Dense(512))

#model= Model(input_sig, output)
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#model.summary()

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.summary()

model.fit(x_train_noisy, x_train_pure,
                epochs=200,
                batch_size=2000,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_pure))

decoded_signals = model.predict(x_test_noisy)

for i in range(24):
    np.savetxt("Prediction_files/Preds_L1_"+str(i)+".txt", np.transpose(decoded_signals[i]))
    
for i in range(24):
    np.savetxt("Pure_signal_files/Preds_L1_"+str(i)+".txt", np.transpose(x_test_pure[i]))

for i in range(24):
    np.savetxt("Noisy_signal_files/Preds_L1_"+str(i)+".txt", np.transpose(x_test_noisy[i]))
    

#np.savetxt("Prediction_files/Preds_GW170817.txt", np.transpose(decoded_signals))
#np.savetxt("Pure_signal_files/Preds_GW170817.txt", np.transpose(x_test_pure))
#np.savetxt("Noisy_signal_files/Preds_GW170817.txt", np.transpose(x_test_noisy))


score = model.evaluate(x_test_noisy, x_test_pure, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])

model.save("model.h5")
print("Saved model to disk")



    
