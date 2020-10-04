# Biomedical Event Extraction as Multi-Turn Question Answering

PEDL is a method for predicting protein-protein assocations from text. The paper describing it will be presented at ISMB 2020.

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

### Generate BioNLP
`./conversion/make_bionlp_data.sh` generates the BioNLP data sets for both PEDL and [comb-dist](https://github.com/allenai/comb_dist_direct_relex/tree/master/relex)

All experiments in the paper have been performed with the masked version of the data, e.g. `distant_supervision/data/BioNLP-ST_2011/train_masked.json`.

### Generate PID
Generating the PID data is a bit more involved:

1. First, we have to download the raw PubMed Central texts: `python download_pmc.py`. CAUTION: This produces over 200 GB of files and spawns multiple processes.
2. Then, we have to download the PubTator Central file (ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz) and place it into the root directory. This file consumes another 80 GB when decompressed.
3. Generate the raw PID data: `./conversion/generate_raw_pid.sh`
4. Generate the final PID data: `./conversion_make_pid.sh`


## Training PEDL

Training can be conducted using the script `scripts/train.sh`.

## Predicting with PEDL
Predictions can be conducted using the script `scripts/predict.sh`.

Corresponding predictions in the `.a*` format can be found in the specified output folder.


## Disclaimer
Note, that this is highly experimental research code which is not suitable for production usage. We do not provide warranty of any kind. Use at your own risk.