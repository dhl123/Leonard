#!/bin/bash
nohup python -u query_withedge_vertex_removeedge_darpa_batch.py -model_lite lite.h5vertex_trace1.hdf5 -model_name LSTM_multi -model vertex_trace1.hdf5 -data_params ../data/vertex_trace1.params.json_s -table_file table_trace1.params.json -dataset_flag 1 -edge_file ../data/edges_trace1.txt -dataset_flag 1 -gpu 0 &




