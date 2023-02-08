import os 
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import json
import struct
import models
import tempfile
import shutil

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-model_path', action='store', dest='model_path',
                    help='model file')
parser.add_argument('-data', action='store', dest='sequence_npy_file',
                    help='data file')
parser.add_argument('-data_params', action='store', dest='params_file',
                    help='params file')
parser.add_argument('-table_file', action='store', dest='table_file',
                    help='table_file')
parser.add_argument('-gpu', action='store', dest='gpu',
                            help='gpu')
args = parser.parse_args()
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
#keras.backend.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def predict_lstm(data_x,data_y, inds,timesteps, alphabet_size, model_name,batch_size):
        X=np.array(data_x)
        Y=data_y
        num_iter=int(len(X)/batch_size)
        prob=[]
        addd=0
        begin1=time()
        if len(X)>batch_size*int(len(X)/batch_size):
            addd=1
        for i in range(num_iter+addd):
         #   begin_time_tmp=time()
            if addd==1 and i==num_iter:
                batch_x=X[batch_size*int(len(X)/batch_size):]
                for inde in range(batch_size-len(batch_x)):
                    batch_x=np.concatenate((batch_x,np.array([X[0]])))
            else:
                batch_x=X[i*batch_size:(i+1)*batch_size]
            batch_x=batch_x.astype(np.float32)
            begin_time_=time()
            interpreter = tf.lite.Interpreter(model_path=args.model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], batch_x)
            interpreter.invoke()
            #print(batch_x)
            prob_tmp = interpreter.get_tensor(output_details[0]["index"])
            #print(prob_tmp)
            end_=time()
            #print(end_-begin_time_)
            for pro in prob_tmp:
                prob.append(pro)
        table_item=[]
        #prob = model.predict(X, batch_size=len(Y))
        for i in range(len(inds)):
            table_item_=[]
            if i ==0:
                begin = 0
                end=inds[i]
            else:
                begin=inds[i-1]
                end=inds[i]
            for j in range(begin,end):
                if np.argmax(prob[j])!=Y[j]:
                    table_item_.append([str(j-begin),str(Y[j])])  
            table_item.append(table_item_)
        #print('each iter time')
        #print(time()-begin1)
        return table_item
def get_slice(data,flag):
    add=0
    if flag==1:
        add=1
    ind=np.where(np.array(data)==flag)[0]
    finaldata=[]
    for i in range(len(ind)):
        if i==0:
            finaldata.append(data[:ind[i]+add])
        else:
            finaldata.append(data[ind[i-1]+1:ind[i]+add])
    return finaldata
def get_slice_index(data,flag):
    ind=np.where(np.array(data)==flag)[0]
    return ind
def translate(data, key=''):
    data_str=''
    for i in data:
        data_str=data_str+id2char_dict[str(i)]
    if key!='':
        data_str=re_values[key][int(data_str)]
    return data_str
from time import *
def main():
        begin_time=time()
        args.temp_dir = tempfile.mkdtemp()
        args.temp_file_prefix = args.temp_dir + "/compressed"
        tf.compat.v1.set_random_seed(42)
        np.random.seed(0)
        series = np.load(args.sequence_npy_file)
        series = series.reshape(-1, 1)        
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(series)
        timesteps = 10
        with open(args.params_file, 'r') as f:
                params = json.load(f)
        params['len_series'] = len(series)
        params['timesteps'] = timesteps
        global id2char_dict
        id2char_dict=params['id2char_dict']
        global char2id_dict
        char2id_dict=params['char2id_dict']
        global re_values
        re_values=params['re_values_dict']
        alphabet_size = len(params['id2char_dict'])+2
        series = series.reshape(-1)
        series=get_slice(series,1)
        data_len=[]
        error_=[]
        order_=[]
        table={}
        batch_size=2048
        batchsize=2048
 #       print('data load time')
 #       print(time()-begin_time)
 #       print(len(series))
        for j in range(int(len(series)/batch_size)):
            timestamp=[]
            data_sub_x=[]
            data_sub_y=[]
            inds=[]
           # print(j*batch_size)
           # if j*batch_size<840000:
                #continue
            for i in range(batch_size):
                timestamp_=get_slice(series[batch_size*j+i],0)[0][8:]
                if series[batch_size*j+i][0]==3:
                    timestamp.append('e:'+translate(timestamp_))
                else:
                    timestamp.append('v:'+translate(timestamp_))
                index_of_time=get_slice_index(series[batch_size*j+i],0)[0]
            #    print(batch_size*j+i)
            #    print(series[batch_size*j+i])
            #    print(index_of_time-timesteps+1)
            #    print('----')
                data_tmp=strided_app(series[batch_size*j+i][index_of_time-timesteps+1:], timesteps+1, 1)
                if i==0:
                    inds.append(len(data_tmp))
                else:
                    inds.append(len(data_tmp)+inds[i-1])
                data_len.append(len(data_tmp))
                for data_index in range(len(data_tmp)):
                    data_sub_x.append(data_tmp[:,:-1][data_index])
                    data_sub_y.append(data_tmp[:,-1:][data_index][0])
            table_item=predict_lstm(data_sub_x,data_sub_y,inds, timesteps, alphabet_size, args.model_name,batchsize)
            for i in range(batch_size):
                error=len(table_item[i])
                if error>0:
                    table[timestamp[i]]=table_item[i]
                    error_.append(error)
                    order_.append(batch_size*j+i)
        timestamp=[]
        data_sub_x=[]
        data_sub_y=[]
        inds=[]
#        print('whole process time')
#        print(time()-begin_time)
        for j in range(int(len(series)/batch_size)*batch_size,len(series)):
            timestamp_=get_slice(series[j],0)[0][8:]
            if series[j][0]==3:
                timestamp.append('e:'+translate(timestamp_))
            else:
                timestamp.append('v:'+translate(timestamp_))
            index_of_time=get_slice_index(series[j],0)[0]
            data_tmp=strided_app(series[j][index_of_time-timesteps+1:], timesteps+1, 1)
            if len(inds)==0:
                inds.append(len(data_tmp))
            else:
                inds.append(len(data_tmp)+inds[j-int(len(series)/batch_size)*batch_size-1])
            data_len.append(len(data_tmp))
            for data_index in range(len(data_tmp)):
                data_sub_x.append(data_tmp[:,:-1][data_index])
                data_sub_y.append(data_tmp[:,-1:][data_index][0])
        table_item=predict_lstm(data_sub_x,data_sub_y,inds, timesteps, alphabet_size, args.model_name,batchsize)
        for i in range(len(series)-int(len(series)/batch_size)*batch_size):
            error=len(table_item[i])
            if error>0:
                table[timestamp[i]]=table_item[i]
                error_.append(error)
                order_.append(i+int(len(series)/batch_size)*batch_size)
        additional_save_time=time()
#        np.save('order_time.npy', order_)
#        np.save('data_len_time.npy', data_len)
#        np.save('error_time.npy',error_)
        print('additional save time')
        print(time()-additional_save_time)
        write_time=time()
        print('write time')
        with open(args.table_file, 'w') as f:
            json.dump(table, f, indent=4)
        end_time=time()
        print(end_time-write_time)
        print('time is')
        print(end_time-begin_time)
        print('read time is')

                                        
if __name__ == "__main__":
        main()

