#!/bin/bash

# model_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/output_models/output_genia"
model_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/output_models/output_tune_seed_1"
data_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/PathwayCuration/BioNLP-ST_2013_PC_development_data"
predictions_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/event_extraction_as_qa/data/"

# Use --task=GE for GENIA

python run/run_mtqa.py \
--task=PC \
--test_data=${data_dir} \
--output_dir=${model_dir} \
--predictions_dir=${predictions_dir} \
--model_name_or_path="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/scibert/scibert_scivocab_uncased" \
--do_predict=True \
--do_lower_case=True \
--fp16=True \
--eval_format_bio_nlp=True
