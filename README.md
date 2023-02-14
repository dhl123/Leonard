# Leonard
Usenix Security'23

## Requirements
Keras: 2.4.3
Tensorflow: 2.4.1
Numpy: 1.19.5


## Storage

```shell
cd src
bash storage.sh
```

The final compressed files are vertex200m.params.json_s.gz (151320 bytes), vertex200m.hdf5.gz (1326020), table200m.params.json.gz (149739), edges200m.txt.gz (207054). The final size is 1.75 MB. The result may be slightly different when runing on different hardware.

## Query:

For both Leonard and other databases, we exclude the database initialization time costs. For weighted searching, download the score map in https://drive.google.com/file/d/18MbTF-5JPWLAmPNlx2-qNwwzySq8dOIo/view?usp=sharing.

```shell
cd src
bash query_trace1.sh
```

## Contact

If you have any questions, please send an email to hailun.ding@rutgers.edu
