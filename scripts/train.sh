#!/bin/bash
model_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/output_models/output"
data_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/PathwayCuration/BioNLP-ST_2013_PC_train_data"
# Use --task=GE for GENIA

python -m torch.distributed.launch --nproc_per_node=4 run/run_mtqa.py \
--task=PC \
--train_data=${data_dir} \
--output_dir=${model_dir} \
--do_train=True \
--overwrite_output_dir=True \
--overwrite_cache=True \
--do_lower_case=True \
--per_gpu_train_batch_size=4 \
--num_train_epochs=16 \
--fp16=True
