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
* Donwload pretrained models for Pathway Curation and GENIA from  

## Training

Training can be conducted using the script `scripts/train.sh`.

## Predicting
Predictions can be conducted using the script `scripts/predict.sh`.

Corresponding predictions in the `.a*` format can be found in the specified output folder.

## Disclaimer
Note, that this is highly experimental research code which is not suitable for production usage. We do not provide warranty of any kind. Use at your own risk.