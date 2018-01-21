# Activity Recognition from Egocentric Photo-Streams

### Introduction

This repository contains the code used in

    @article{paa2017,
       author = {A. Cartas and J. Marin and P. Radeva and M. Dimiccoli},
       title = {Recognizing Daily Activities from Egocentric Photo-Streams},
       journal = {ArXiv e-prints},
       year = 2017
    }

### Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Downloads](#downloads)

### Installation

1. Clone this repository
  	```Shell
  	git clone --recursive https://github.com/gorayni/egocentric_photostreams.git

2. Download the NTCIR-12 dataset at http://ntcir-lifelog.computing.dcu.ie/NTCIR12/.

3. Create a symbolic link datasets/ntcir/images pointing to NTCIR_Lifelog_formal_run_Dataset/NTCIR-Lifelog_formal_run_images.

4. Split the data by executing the *Dataset split* notebook. This will create the cross-validation splits in the directory *data*.

## Static Image Classification

### Training

Once the dataset was split, then the models can be trained by

```bash
python training/train_cnn.py --network=vgg-16 --data_dir=data/static --weights_dir=weights/vgg-16

python training/train_rf.py --network=vgg-16 --data_dir=data/static --weights_dir=weights/vgg-16 -l 'predictions' 'fc1' 
python training/train_rf.py --network=vgg-16 --data_dir=data/static --weights_dir=weights/vgg-16 -l 'fc1'
python training/train_rf.py --network=vgg-16 --data_dir=data/static --weights_dir=weights/vgg-16 -l 'fc2' 
python training/train_rf.py --network=vgg-16 --data_dir=data/static --weights_dir=weights/vgg-16 -l 'fc1' 'fc2'


python training/train_cnn.py --network=resNet50 --data_dir=data/static --weights_dir=weights/resNet50
python training/train_rf.py --network=resNet50 --data_dir=data/static --weights_dir=weights/resNet50 -l 'flatten_1'

python training/train_cnn.py --network=inceptionV3 --data_dir=data/static --weights_dir=weights/inceptionV3
python training/train_rf.py --network=inceptionV3 --data_dir=data/static --weights_dir=weights/inceptionV3 -l 'global_average_pooling2d_1' 
```

### Testing

```bash
python testing/test_cnn.py --network=vgg-16 --data_dir=data/static --results_dir=results/vgg-16 --weights_dir=weights/vgg-16 

python testing/test_rf.py --network=vgg-16 --data_dir=data/static --results_dir=results/vgg-16 --weights_dir=weights/vgg-16 --layer 'predictions' 'fc1'
python testing/test_rf.py --network=vgg-16 --data_dir=data/static --results_dir=results/vgg-16 --weights_dir=weights/vgg-16 --layer 'fc1'
python testing/test_rf.py --network=vgg-16 --data_dir=data/static --results_dir=results/vgg-16 --weights_dir=weights/vgg-16 --layer 'fc2'
python testing/test_rf.py --network=vgg-16 --data_dir=data/static --results_dir=results/vgg-16 --weights_dir=weights/vgg-16 --layer 'fc1' 'fc2'

python testing/test_cnn.py --network=resNet50 --data_dir=data/static --results_dir=results/resNet50 --weights_dir=weights/resNet50

python testing/test_rf.py --network=resNet50 --data_dir=data/static --results_dir=results/resNet50 --weights_dir=weights/resNet50 --layer 'flatten_1'

python testing/test_cnn.py --network=inceptionV3 --data_dir=data/static --results_dir=results/inceptionV3 --weights_dir=weights/inceptionV3

python testing/test_rf.py --network=inceptionV3 --data_dir=data/static --results_dir=results/inceptionV3 --weights_dir=weights/inceptionV3 -l 'global_average_pooling2d_1'

```