#!/bin/bash

# PathwayCuration
gold_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/PathwayCuration/BioNLP-ST_2013_PC_development_data"
predictions="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/event_extraction_as_qa/data/PC/*.a2"

source activate tees-env

python baseline/evaluation-PC.py ${predictions} \
--ref_directory=${gold_dir}

# GENIA
# gold_dir="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/Genia11/BioNLP-ST_2011_genia_devel_data_rev1"
# predictions="/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/event_extraction_as_qa/data/GE/*.a2"

# ./baseline/a2-evaluate.pl -g ${gold_dir} -sp -t1 ${predictions}
