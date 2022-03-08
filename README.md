# PET

This repo contains the training and evaluation code for the paper [Prior Knowledge and Memory Enriched Transformer for Sign Language Translation].

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## The reproduction of results
Please download the pre-trained model in the place and put the model in the folder `model`.
And excute the script, the results may be a little different from the results reported in the paper. 

  `python -m signjoey test configs/sign.yaml`.

## The training of the method

   (1) Firstly, to execute the following script, please comment the the following lines ''train.py (lines 207-212, 1046-1048), decoders.py (lines 666-667, 603-604)''
  `python -m signjoey train configs/sign.yaml`
  
   (2) Second, remove the comments of ''train.py (lines 207-212, 1046-1048)'', add comments for ''decoders.py (lines 665, 667, 603, 605)'', execute the following command,
  `python -m signjoey train configs/sign.yaml`

## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2020sign,
      author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
      booktitle = {CVPR},
      year = {2020}
    }

## Acknowledgements
<sub>This work was supported in part by the National Key R\&D Program of China under Grant No.2020YFC0832505, National Natural Science Foundation of China under Grant No.61836002, No.62072397 and Zhejiang Natural Science Foundation under Grant LR19F020006.</sub>
