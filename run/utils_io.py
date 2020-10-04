""" Utility IO functions for run_mtqa.py """

from __future__ import absolute_import, division, print_function
import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import logging
import random
import numpy as np
import torch
import fractions

from torch.utils.data import TensorDataset
from sklearn import preprocessing
from run.utils_ner import (generate_examples, convert_examples_to_features)
from datastructures.datatypes import LAMBDA_WEIGHTS

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, data_dir, recursion_depth, question_type,
                            prior_events=None, evaluate=False, cache=True):
    ''' Load and cache new datasets '''

    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    question_number = 2 * recursion_depth + question_type
    cached_features_file = os.path.join(
        root_path + "/data/cached_features", "cached_{}_{}_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length), question_number, args.task))
    cached_features_file_next = os.path.join(
        root_path + "/data/cached_features", "cached_{}_{}_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length), question_number + 1, args.task))
    cached_label_encoder = os.path.join(
        root_path + "/data/cached_features", "classes_{}_{}_{}.npy".format(mode, question_number, args.task))
    cached_file_names_encoder = os.path.join(
        root_path + "/data/cached_features", "classes2_{}_{}_{}.npy".format(mode, question_number, args.task))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        subject_label_encoder = preprocessing.LabelEncoder()
        subject_label_encoder.classes_ = np.load(cached_label_encoder)
        file_names_encoder = preprocessing.LabelEncoder()
        file_names_encoder.classes_ = np.load(cached_file_names_encoder)
        if os.path.exists(cached_features_file_next):
            new_answers = True
        else:
            new_answers = False
    else:
        logger.info("Creating features from dataset file at {} for {}_{}".format(data_dir, mode, question_number))
        examples, subjects_list, file_names, new_answers = generate_examples(
            data_dir, mode, recursion_depth, question_type, prior_events, args.do_predict)
        features, subject_label_encoder, file_names_encoder = convert_examples_to_features(
            examples, labels, args.max_seq_length, args.doc_stride, tokenizer,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0, pad_token_label_id=pad_token_label_id,
            subject_list=subjects_list, file_names=file_names)
        if args.local_rank in [-1, 0]:
            # Don't save dynamic features from questions building on previous predictions
            if cache:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
                np.save(cached_label_encoder, subject_label_encoder.classes_)
                np.save(cached_file_names_encoder, file_names_encoder.classes_)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    whitespace_bools = torch.tensor([f.whitespace_bools for f in features], dtype=torch.bool)
    position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)
    token_is_max_context = torch.tensor([f.token_is_max_context for f in features], dtype=torch.long)
    subjects = torch.tensor([f.subjects for f in features], dtype=torch.long)
    pubmed_ids = torch.tensor([f.pubmed_id for f in features], dtype=torch.long)
    question_ids = torch.tensor([f.question_id for f in features], dtype=torch.long)
    question_types = torch.tensor([question_number for f in features], dtype=torch.long)
    subject_lengths = torch.tensor([f.subject_length for f in features], dtype=torch.long)
    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids, token_is_max_context, subjects, pubmed_ids, whitespace_bools,
                            position_ids, question_ids, question_types, subject_lengths)
    return dataset, subject_label_encoder, file_names_encoder, new_answers


def save_model(args, global_step, model):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--train_data", default=root_path + "/data/iob/PathwayCuration/train",
                        type=str, help="The train data directory.")
    parser.add_argument("--dev_data", default=root_path + "/data/iob/PathwayCuration/dev_sample",
                        type=str, help="The dev data directory.")
    parser.add_argument("--test_data", default=root_path + "/data/iob/PathwayCuration/test",
                        type=str, help="The test data directory.")
    parser.add_argument("--model_name_or_path", default=root_path + "/scibert/scibert_scivocab_uncased",
                        type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=root_path + "/output_dir", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--predictions_dir", default=root_path + "/predictions", type=str,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--task", default="PC", type=str,
                        help="Task to evaluate on.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--do_train", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--get_predictions", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Use predictions or debug groundtruth")
    parser.add_argument("--debug", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Log debugging infos or not")
    parser.add_argument("--first", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Model only on first question trained")
    parser.add_argument("--eval_format_bio_nlp", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Evaluation in a question answering format")
    parser.add_argument("--weights", default=LAMBDA_WEIGHTS, type=lambda x:
                        [float(fractions.Fraction(item)) for item in x.split(',')],
                        help="Lambda Weights for all six questions denoted as fractions")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--wandb", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Loggint with Weight and biases.")
    parser.add_argument("--logging_steps", type=int, default=149,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending "
                             + "and ending with step number")
    parser.add_argument("--no_cuda", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--write_predictions", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Write predictions in a .iob file")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    return parser
