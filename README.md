# Biomedical Event Extraction as Multi-Turn Question Answering

## Requirements
* `python >= 3.7`
* `pip install -r requirements.txt`

## Download Files

### Data
* Download Pathway Curation training, development and test sets from http://2013.bionlp-st.org/tasks/.
* Download GENIA11 training, development and test sets from http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/downloads.shtml.

### Miscellaneous
* Download pretrained SciBert checkpoint [scibert-scivocab-uncased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar).
* Donwload pretrained models for Pathway Curation and GENIA from [here](https://drive.google.com/file/d/1JuQJGu5V3AH12WHApC4_kiuzwiKv9JdA/view?usp=sharing). Extract an unzip *.tar.gz files.

## Training

Training can be conducted using the script `scripts/train.sh`. Change model directory `model_dir` and data directory `data_dir` to the respective locations in your systems. For further options, refer to the file /run/utils_io.py.

## Predicting

Predictions can be conducted using the script `scripts/predict.sh`. Change model directory `model_dir` and data directory `data_dir` to the respective locations in your systems. Predictions in the BioNLP `.a*` format are written to the specified output folder `predictions_dir`. For further options, refer to the file /run/utils_io.py.

## Disclaimer
Note, that this is highly experimental research code which is not suitable for production usage. We do not provide warranty of any kind. Use at your own risk.