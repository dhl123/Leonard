import os 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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
import json
import struct
import models
import tempfile
import shutil
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-dataset_flag', action='store', dest='dataset_flag')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-gpu', action='store', dest='gpu',
                    help='gpu')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-data_params', action='store', dest='params_file',
                    help='params file')
parser.add_argument('-table_file', action='store', dest='table_file',
                    help='table_file')
parser.add_argument('-model_lite', action='store', dest='model_lite',
                    help='table_file')
parser.add_argument('-edge_file', action='store', dest='edge_file',
                    help='table_file')
args = parser.parse_args()
from keras import backend as K
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(sess)
from time import *
def translate_edge(path):
    f=open(path,'r',encoding="unicode_escape")
    strr=f.read()
    strr=strr.split('\n')
    edges=[]
    edges.append(eval(strr[0]))
    edges.append(eval(strr[1]))
    edges.append(eval(strr[2]))
    edges.append(eval(strr[3]))
    return edges
def strided_app(a, L, S):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)
global key_pattern
with open(args.params_file, 'r') as f:
    params = json.load(f)
global id2char_dict
id2char_dict=params['id2char_dict']
global char2id_dict
char2id_dict=params['char2id_dict']
global re_values
re_values=params['re_values_dict']
global mins
mins=params['mins']
global key_template_dict
key_template_dict=params['key_template_dict']
global key_template
key_template={}
for i in key_template_dict.keys():
    key_template[key_template_dict[i]]=i
global alphabet_size
alphabet_size = len(params['id2char_dict'])+2
def get_for_presict(queries,timesteps):
    tmp_query=[]
    for i in range(len(queries)):
        tmp_query.append(np.array(queries[i][len(queries[i])-timesteps:]))
    return tmp_query
def generate_query_counters(query_sequence,counter,flag):
    sentences=[]
    counters=[]
    strr=[]
    if flag==-1:
        for i in range(len(query_sequence)):
            strr.append('verteid:')
    if flag==-2:
        for i in range(len(query_sequence)):
            strr.append('eventid:')
    if flag>-1:
        for i in range(flag+1):
            strr.append('verteid:')
        for i in range(flag+1,len(query_sequence)):
            strr.append('eventid:')
    for i in range(len(query_sequence)):
        tmp_sentence=[]
        tmp_strr=query_sequence[i]
        counters.append(str(query_sequence[i]))
        for j in strr[i]:
            tmp_sentence.append(int(char2id_dict[j]))
        for j in str(tmp_strr):
            tmp_sentence.append(int(char2id_dict[j]))
        tmp_sentence.append(0)
        sentences.append(tmp_sentence)
    return sentences,counters
def predict_lstm(queries,counters_,timesteps, alphabet_size,start,overlen,overflag):
        flag=[]
        queries_len=len(queries)
        for i in range(len(queries)):
            flag.append(False)
        strr=[]
        if overflag!=-1:
            for i in range(overflag,len(queries)):
                flag[i]=True
        for i in range(len(queries)):
            if flag[i]:
                continue
            else:
                if queries[i][0]==3:
                    strr.append('e:')
                else:
                    strr.append('v:')
        begin=0
        while False in flag:
            lenn=0
            record=[]
            tmp_query=get_for_presict(queries,timesteps)
            begin_=time()
            X=np.array(tmp_query).astype(np.float32)
            interpreter = tf.lite.Interpreter(model_path=args.model_lite)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], X)
            interpreter.invoke()
            prob = interpreter.get_tensor(output_details[0]["index"])
            for i in range(len(queries)):
                if flag[i]:
                    continue
                if strr[i]+counters_[i] in table.keys():
                    if str(begin) in table[strr[i]+counters_[i]][0]:
                        queries[i].append(int(table[strr[i]+counters_[i]][1][table[strr[i]+counters_[i]][0].index(str(begin))]))
                    else:
                        queries[i].append(np.argmax(prob[i]))
                else:
                    queries[i].append(np.argmax(prob[i]))
                if queries[i][-1]==1:
                    flag[i]=True
            begin=begin+1
        return queries
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
def translate_str(data,key=''):
    process=''
    if isinstance(data,list):
        for i in data:
            process=process+id2char_dict[str(i)]        
    else:
        process=process+id2char_dict[str(data)]
    return process
def translate(data, key='',flag=0):
    data=data[0:len(data)-1]
    if data[0]==3:
        flag=1
    else:
        flag=0
    data=get_slice(data,0)   
    counter_tmp=data[0][8:]
    key_index=data[1]
    strrr=''
    key_list={}
    for i in key_index:
        strrr=strrr+id2char_dict[str(i)]
    key_index=int(strrr)
    key_pattern=key_template[key_index].split(',')
    if flag==0:
        key_list['hash']=translate_str(counter_tmp)
    begin=2
    for index in range(len(key_pattern)):
        key=key_pattern[index]
        if key!='hash':
            if key=='parentVertexHash':
                key_list[key]=str(o1[int(translate_str(counter_tmp))])
                continue
            elif key=='childVertexHash':
                key_list[key]=str(o2[int(translate_str(counter_tmp))])
                continue
            elif key=='timestampNanos':
                key_list[key]=str(int(translate_str(data[begin],key))+mins[0])
            elif key=='startTimestampNanos':
                key_list[key]=str(int(translate_str(data[begin],key))+mins[1])
            elif key=='sequence':
                key_list[key]=str(int(re_values[key][int(translate_str(data[begin],key))])+mins[2])
            else:
                key_list[key]=re_values[key][int(translate_str(data[begin],key))]
            begin=begin+1
    return key_list
def process_table(table):
    table_tmp={}
    for key in table.keys():
        table_tmp[key]=[]
        table_tmp[key].append([])
        table_tmp[key].append([])
        for i in range(len(table[key])):
            table_tmp[key][0].append(table[key][i][0])
            table_tmp[key][1].append(table[key][i][1])
    return table_tmp
from time import *
global table 
with open(args.table_file, 'r') as f:
    params = json.load(f)
table=process_table(params)
def iter_bfs_score(now,sett_edges,sett_points,G,mapp,data):
    searched=set()
    count=0
    #now=sorted(now, key=lambda x: x[3])
    while now:
        large=-1.0
        tmp_edge=0
        for j in now:
            if large<float(j[3]):
                tmp_edge=j
                large=float(j[3])
            elif large==float(j[3]):
                if data[int(tmp_edge[0])+1][0]<=data[int(j[0])+1][0]:
                    tmp_edge=j
            else:
                continue
        now.remove(tmp_edge)
        i= tmp_edge
        sett_points.add(i[2])
        sett_points.add(i[1])
        sett_edges.add(i[0])
        if (len(sett_points)+len(sett_edges))>4096:
            break
        if i[1] in searched:
            continue
        idx = mapp[i[1]]
        for target in G[idx]:
            if target[1] not in sett_edges:
                now.append((target[1],target[0],i[1],target[2]))
        searched.add(i[1])

import csv  
nodes_for_dataset=[[1099876,672640,1683629,514060,468934,1787378,495914,1673209,1854115,499163,1662376,1058801,692629,1339861,87517,423047,351501,1624091,695449,75339,1634507,791085,1616974,1612807,1639534,718370,1600687,702072,1807501,1304266,1360273,478092,284379,44238,569243,1132880,922447,2019306,140756,1009252,1506758,1191750,318206,674460,684007,1530136,684763,254500,1178172,364387,133153,1116597,673242,360253,1640513,852603,272086,1485387,592571,1230191,1829617,1189368,2015994,1578470,997119,1113838,1870472,1924379,1082974,792123,1736645,1554034,456956,1449898,172122,1822603,2085431,664535,324110,518400,291993,2027534,2069026,2026525,1819327,1435847,197891,146763,672947,1728040,1582844,1470589,979480,1034243,1642994,1369251,1506015,1506378,1064841,1527819],[386380,1419319,489746,1208589,1351369,1322994,1441152,1457591,831647,1019033,1000324,1619080,1536855,1650251,1235636,1219451,412740,1502708,1775852,184512,403005,1098833,45920,1659365,1340094,574275,217646,1773816,1463021,1497852,2026509,499354,1612677,739469,1509590,1806135,433571,1851742,989442,933631,1563975,805993,1113467,1475433,1687174,66988,1500081,374387,1549762,479145,119387,38651,157823,643752,1366003,799440,270628,118770,464675,1650723,119396,2016400,2019577,1499278,159201,1435211,1298889,1788413,1123153,1327216,1649102,1991089,185710,1758935,1650063,1128066,1628617,1371920,1875922,321859,1466785,2043339,1329074,476021,1029566,2077727,1298458,619837,1120426,1424348,518053,389895,1058097,215756,1217221,859212,1497905,687354,810243,2085209],[2054723,1172632,1485986,1549171,1058520,2176463,1212923,1455768,478597,836553,2285074,2223302,2051109,1898061,1206031,146034,757153,1409430,1184120,1277486,1055475,1381211,354411,1542675,547630,503181,977851,863289,602619,139201,626089,2029164,523411,980114,363525,2279184,1859295,951357,1763616,893035,1187182,1728043,2477699,2070710,254372,2121434,407311,1491560,315540,922657,248777,1799208,1127736,103205,437186,1657621,744882,1536181,1475247,2144395,1175033,1677416,1984768,1719981,112217,1954096,694702,1704130,1564996,725601,580634,2097648,1498115,1342585,554636,1358709,1397009,334768,2025087,2132241,2254621,122438,2369066,1920374,1003936,1563641,2303017,2112770,108813,1342015,2451765,277774,463982,2320394,662901,1372130,1873686,1327713,11605,1493703],[2028413,1537437,2307752,750488,12411,1457486,67400,919108,1706665,1721297,1355245,1710035,981014,2061880,786164,1296562,1888900,1408198,1689322,1271718,2256451,2152657,386670,440839,1457064,311472,371297,1063072,1089877,2228589,749696,1570158,647429,306897,2328721,1572136,114364,945255,1354622,1688894,863135,799935,770571,1241480,2093111,1606483,187486,2257951,689875,615366,1418675,1529873,1405346,169461,659628,795627,2212837,2381089,1481331,1229565,427972,1562650,2304459,16150,736761,592786,1673849,2324251,2368891,430015,343653,1309356,1564634,621348,1348223,1500490,1550379,2043612,1298832,35512,1464280,496359,1176916,288298,641753,2045693,2012950,357783,722411,1017849,779582,1533708,449980,2074509,116983,88608,795783,643371,1107616,745769],[1661382,1650724,1269905,526014,114253,1693696,1629943,1169519,1321876,1726021,1693831,1768043,1157268,1332937,847821,300570,624866,1628081,1228962,825931,113235,1455527,1584311,911150,770295,734539,296826,1310022,1697164,1218926,1410931,1762553,1117447,1412238,1144590,270948,1190897,1311171,1366439,470059,747390,705080,1736835,1288862,959894,1193369,1156745,1747874,324553,1365218,25873,950816,419235,1103630,1713625,1715518,192929,111477,1377550,1069384,765409,942147,1208167,604624,1033381,1635313,656383,1453264,975508,861160,1357715,1783658,562905,879057,98371,1402400,672634,231872,155267,1223763,1230825,752588,898699,1278801,1366493,1328788,229544,291064,1351109,1681764,127579,1293580,1551511,1294815,458605,1133961,784528,747790,393502,172665]]
def main():
        start_nodes=nodes_for_dataset[int(args.dataset_flag)-1]#[672640,1683629,514060,468934,1787378,495914,1673209,1854115,499163,692629]
        edges=translate_edge(args.edge_file)
        global o1
        o1=edges[3]
        global o2
        o2=edges[2]
        f=open('/src/workspace/for_spade_db_data/score_trace'+str(args.dataset_flag)+'.txt')
        score_str=f.read()
        score_str=score_str.split("\n")
        score_map={}
        for i in score_str:
            tmp_str=i.split(";")
            score_map[tmp_str[0]]=float(tmp_str[1])
        data=[]
        with open('/src/workspace/edge_trace'+str(args.dataset_flag)+'.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            for i in reader:
                data.append(i)
        mapp = {}
        G = []
        for i in range(len(o1)):
            a,b = o1[i],o2[i]
            if a not in mapp :
                mapp[a] = len(G)
                G.append([])
            if b not in mapp:
                mapp[b] = len(G)
                G.append([])
            G[mapp[a]].append((b,i,score_map.get(data[i+1][0])))
        for start in start_nodes:
            begin_time=time()
            now=[]
            sett_edges = set()
            sett_points = set()
            indexx=[]
            counter=[]
            for i in range(len(o1)):
                if int(o1[i])==start:
                    indexx.append(i)
            for i in indexx:
                now.append((i,o2[i],o1[i],score_map.get(data[i+1][0])))
            iter_bfs_score(now,sett_edges,sett_points,G,mapp,data)
            ver_len=len(sett_points)
            
            query=[]
            for ver in sett_points:
                query.append(ver)
            for ed in sett_edges:
                query.append(ed)
            end_time1=time()
            tf.compat.v1.set_random_seed(42)
            np.random.seed(0)     
            timesteps = 10  
            total_counter=len(query)
            batch_size=2048
            final_results=[]
            final_sentences=[]
            addd=0
            if total_counter>int(total_counter/batch_size)*batch_size:
                addd=1
            for j in range(int(total_counter/batch_size)+addd):
                if addd==1 and j==int(total_counter/batch_size):
                    flag=-1
                    if batch_size*j<ver_len:
                        flag=ver_len-1-batch_size*j
                        begin1=flag
                    else:
                        flag=-2
                    queries,counters_=generate_query_counters(query[batch_size*j:],counter,flag)
                    tmp_query_len=len(queries)
                    tmp_query,tmp_counter=generate_query_counters(query[:2],counter,-1)
                    for i in range(batch_size-tmp_query_len):
                        queries.append(tmp_query[0])
                        counters_.append(tmp_counter[0])
                    final_sentences=predict_lstm(queries,counters_,timesteps, alphabet_size,batch_size*j,ver_len,tmp_query_len)
                    final_sentences=final_sentences[:tmp_query_len]
                else:
                    flag=-1
                    if batch_size*j<=(ver_len-1)<=batch_size*(j+1):
                        flag=ver_len-1-(batch_size*j)
                        begin1=flag
                    if (ver_len-1)<batch_size*j:
                        flag=-2
                    queries,counters_=generate_query_counters(query[batch_size*j:batch_size*(j+1)],counter,flag)
                    final_sentences=predict_lstm(queries,counters_,timesteps, alphabet_size,batch_size*j,ver_len,-1)
                for i in range(len(final_sentences)):
                    final_results.append(translate(final_sentences[i],flag=0))
            final_result=[]
            for i in range(len(final_results)):
                final_result.append(str(final_results[i]))
            end_time=time()
            edges_output=[]
            vertex_output=[]
            pidd={}
            count_for=[]
            print(start)
            print('searched number')
            print(len(sett_edges),len(sett_points))
            print(end_time1-begin_time)
            print(end_time-end_time1)
if __name__ == "__main__":
        main()
