
import json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
#from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
# typical train
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from sklearn.model_selection import train_test_split

import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
tf.compat.v1.experimental.output_all_intermediates(True)
import models
from time import *
begin_time_train=time()
#tf.set_random_seed(42)
#np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store', default=None,
                    dest='data',
                    help='choose sequence file')
parser.add_argument('-test', action='store', default=None,
                    dest='testdata',
                    help='choose sequence file')
parser.add_argument('-gpu', action='store', default="2",
                    dest='gpu',
                    help='choose gpu number')
parser.add_argument('-name', action='store', default="model1",dest="name")
parser.add_argument('-model', action='store', default="model",
                    dest='model',
                    help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,
                    dest='model_name',
                    help='name of the model to call')
parser.add_argument('-log_file', action='store',
                    dest='log_file',
                    help='Log file')
parser.add_argument('-batchsize', action='store',
                    dest='batchsize',
                    help='batchsize')
parser.add_argument('-epoch', action='store',
                    dest='epoch',
                    help='epoch', default=4)
parser.add_argument('-param', action='store',dest='param')
parser.add_argument('-lr', action='store',dest='lr',default=0.001)
import keras.backend as K
tf.compat.v1.disable_eager_execution()
def loss_fn(y_true, y_pred):
        return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

def strided_app(a, L, S): 
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_single_output_data(file_path,batch_size,time_steps):
        series = np.load(file_path)
        series = series.reshape(-1, 1)
        series_ = series.reshape(-1)
        #for i in range(27):
        #    if i not in series_:
        #        print(i)
        #        series=np.append(series,np.array([[i]]),axis=0)
        series_=0
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(series)
        series = series.reshape(-1)
        return series,onehot_encoder
        print("TTTTTTTTTTTTTTTTTTTTTTTTT")
        return X,Y
def process_data(serie,onehot_encoder,batch_size,time_steps):
    data = strided_app(serie, time_steps+1, 1)
    data1=np.copy(data)
    np.random.shuffle(data1)
    X = data1[:, :-1]
    Y = data1[:, -1:]
    Y = onehot_encoder.transform(Y)
    return X,Y
def get_slice(data,flag):
    ind=np.where(np.array(data)==flag)[0]
    return ind

def generator(avoid,serie,onehot_encoder,batchsize,sequence_length):
    while True:
        ret=[]
        start=0
        print(onehot_encoder)
        for i in range(sequence_length,len(serie)):
            if i == avoid[start] and start<len(avoid)-1:
                start=start+1
            else:
                ret.append(i)
        choice=np.random.choice(ret, len(ret), replace=False)
        choice=choice[:int(len(choice)/batchsize)*batchsize]
        num_batch=int(len(choice)/batchsize)
        choice=choice.reshape(num_batch,batchsize)
        for i in range(num_batch):
            data=[]
            for j in choice[i]:
                data.append(serie[j-sequence_length:j+1])
            data=np.array(data)
            X_tmp = data[:, :-1]
            Y_tmp = data[:, -1:]
            Y_tmp = onehot_encoder.transform(Y_tmp)
            yield (X_tmp,Y_tmp)

def fit_model(avoid,raw_data,onehot_encoder,bs, sequence_length,nb_epoch, model):
        optim = keras.optimizers.Adam(lr=float(arguments.lr), beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
        model.compile(loss=loss_fn, optimizer=optim,metrics=['accuracy'])
        #model.load_weights(arguments.model)#remove if not finetune
        checkpoint = ModelCheckpoint(arguments.name, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0, patience=3, verbose=1)
        callbacks_list = [checkpoint, csv_logger, early_stopping]
        model.fit_generator(generator(avoid,raw_data,onehot_encoder,bs,sequence_length),steps_per_epoch=int((len(raw_data)-sequence_length+1-len(avoid))/batch_size),epochs=nb_epoch, use_multiprocessing=True,shuffle=True, verbose=1, callbacks=callbacks_list)
        model.save("lite.h5")                
                
arguments = parser.parse_args()
print(arguments)
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
#keras.backend.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)
batch_size=int(arguments.batchsize)
sequence_length=10
num_epochs=int(arguments.epoch)
with open(arguments.param, 'r') as f:
    params = json.load(f)
alphabet_size = len(params['id2char_dict'])+2
print(alphabet_size)
raw_data,onehot_encoder= generate_single_output_data(arguments.data,batch_size, sequence_length)

ind=get_slice(raw_data,1)
avoid=[]
for i in range(len(ind)):
    # strr=[]
    # begin=0
    # end=0
    # if i==0:
    #     begin=-1
    #     strr=raw_data[:ind[0]+1]
    # else:
    #     begin=ind[i-1]
    #     strr=raw_data[ind[i-1]+1:ind[i]+1]
    # time_index=get_slice(strr,0)[1]
    # print(strr)
    # print(time_index)
    # for index_ in range(time_index):
    #     avoid.append(index_+1+begin)
    for index_ in range(sequence_length):
        avoid.append(ind[i]+index_+1)
model = getattr(models, arguments.model_name)(batch_size, sequence_length, alphabet_size)
fit_model(avoid,raw_data,onehot_encoder,batch_size,sequence_length,num_epochs,model)
print(time()-begin_time_train)
