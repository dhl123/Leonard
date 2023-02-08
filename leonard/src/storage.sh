#!/bin/bash  
python -u ../data/parse_vertex_ef_removeedge_pid.py -edge_file ../data/edges200m.npy -input_path ../raw_data/vertex200m.csv -input_path1 ../raw_data/edge200m.csv -output ../data/vertex200m.npy -param ../data/vertex200m.params.json
wait
python simplify_params.py -params vertex200m.params.json
wait
rm vertex200m.params.json
wait
python -u trainer_test_darpa.py -d ../data/vertex200m.npy -epoch 15 -batchsize 4096 -model_name LSTM_multi -name vertex200m.hdf5 -log_file ../data/logs_data/FC.log.csv -param ../data/vertex200m.params.json_s
wait
python transfer_edge_to_txt.py
wait
python only_lite_convert.py -model vertex200m.hdf5
wait
python -u check_prediction_time.py -gpu 0 -model vertex200m.hdf5 -model_name LSTM_multi -data ../data/vertex200m.npy -data_params ../data/vertex200m.params.json_s -model_path lite.h52048vertex200m.hdf5 -table_file table200m.params.json
wait

gzip vertex200m.hdf5
wait
rm lite.h52048vertex200m.hdf5
wait
rm ../data/vertex200m.npy
wait
gzip ../data/vertex200m.params.json_s
wait
gzip table200m.params.json
wait
rm ../data/edges200m.npy
wait
gzip ../data/edges200m.txt
