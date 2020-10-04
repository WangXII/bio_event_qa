# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
    Runs biomedical event extraction as multi-turn question answering.
    Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_ner.py.
'''

from __future__ import absolute_import, division, print_function
import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import pickle
import glob
import logging
import wandb
import numpy as np
import torch

from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from run.utils_run import update_with_nn_output, initialize_loaders, update_metadata, extract_from_tensor_dataset, extract_from_nn_output, train_example
from run.utils_io import load_and_cache_examples, save_model, set_seed, parse_arguments
from run.bert_model import BertForQuestionAnswerTagging
import datastructures.datatypes as datatypes
from metrics.sequence_labeling import precision_score, recall_score, f1_score, classification_report
# from baseline.evaluate_questions import convertDirectory, convertOutput, evaluateAllQuestions
from data_processing.build_events_bionlp import convertOutputToAStarFormat

logger = logging.getLogger(__name__)


def train(args, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """

    # Generate examples for all questions
    train_datasets = []
    recursion_depth = 0
    question_type = 0
    new_answers = True
    prior_events = None
    while new_answers is True:
        question_number = 2 * recursion_depth + question_type
        dataset, subject_label_encoder, file_names_encoder, tmp_new_answers = load_and_cache_examples(
            args, tokenizer, labels, pad_token_label_id, "train", args.train_data, recursion_depth, question_type, prior_events)
        if question_type == 0:
            new_answers = tmp_new_answers
        # if recursion_depth == 4 and question_type == 0:
        #     logger.info(new_answers)
        if new_answers:
            train_datasets.append(dataset)
            logger.info(" Question {} number examples = {}".format(question_number, len(dataset)))
        if question_type == 0:  # Use prior_events from trigger question, TODO: Change this to also include arguments in prior_events
            prior_events = extract_from_tensor_dataset(dataset, subject_label_encoder, file_names_encoder, labels, tokenizer)
        if question_type == 0:
            question_type += 1
        elif question_type == 1:
            question_type = 0
            recursion_depth += 1
        if args.first is True:  # Compare multi-turn task vs training on single question
            new_answers = False

    train_dataset = ConcatDataset(train_datasets)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader, scheduler = initialize_loaders(args, train_dataset, optimizer)

    weights = args.weights
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for iteration in train_iterator:
        logger.info("  Iteration = %d", iteration)
        model.train()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            tr_loss, global_step = train_example(args, model, step, batch, optimizer, scheduler, amp, tr_loss, global_step, weights, iteration)
            # Save after an epoch
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, global_step, model)

        # Evaluate the training results
        if args.local_rank in [-1, 0] and args.evaluate_during_training:
            evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="train", data_dir=args.train_data,
                     prefix=global_step, train_model=True, reports=False, wandb_log=True)
            evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", data_dir=args.dev_data,
                     prefix=global_step, train_model=True, reports=False, wandb_log=True)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, prefix="", train_model=False, reports=False, wandb_log=False):

    # Generate answers for all questions
    weights = args.weights
    answer_tuples = []
    recursion_depth = 0
    prior_events = None
    eval_loss = 0.0
    new_events, prior_events_bool, loss = evaluate_question(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, recursion_depth, 0,
                                                            prior_events, weights, "", train_model, reports, wandb_log)
    eval_loss += loss
    answer_tuples.append(new_events)
    # logger.info(prior_events_bool)
    while prior_events_bool is True:
        prior_events = new_events
        new_events, _, loss = evaluate_question(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, recursion_depth, 1,
                                                prior_events, weights, "", train_model, reports, wandb_log)
        eval_loss += loss
        if new_events:  # Check if dict empty
            answer_tuples.append(new_events)
        recursion_depth += 1
        new_events, prior_events_bool, loss = evaluate_question(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, recursion_depth, 0,
                                                                prior_events, weights, "", train_model, reports, wandb_log)
        eval_loss += loss
        if new_events:  # Check if dict empty
            answer_tuples.append(new_events)
    logger.info("Eval_Loss_Total: " + str(eval_loss))

    if args.eval_format_bio_nlp:  # Save question answer tuples in .a2-files of standard BioNLP formats
        with open(args.predictions_dir + "event_tuples.npy", 'wb') as f:
            pickle.dump([args] + answer_tuples, f)
        convertOutputToAStarFormat(args, data_dir, answer_tuples)


def evaluate_question(args, model, tokenizer, labels, pad_token_label_id, mode, data_dir, recursion_depth, question_type, prior_events=None,
                      lambda_weights=[], prefix="", train_model=False, reports=False, wandb_log=False):
    """ Evaluate one question type. """

    cache = not args.get_predictions if not (recursion_depth == 0 and question_type == 0) else True
    question_number = 2 * recursion_depth + question_type
    eval_dataset, subject_label_encoder, file_names_encoder, _ = load_and_cache_examples(
        args, tokenizer, labels, pad_token_label_id, mode, data_dir, recursion_depth, question_type, prior_events, evaluate=True, cache=cache)

    # multi-gpu evaluate, if using train_model, model is already parallelized!
    if args.n_gpu > 1 and not train_model:
        model = torch.nn.DataParallel(model)
    model.eval()
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # logger.info("***** Running evaluation %s *****", prefix)
    # logger.info("  Question %s! ", question_number)
    # logger.info("  Num examples = %d", len(eval_dataset))

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = out_label_ids = out_token_ids = is_max_token = subjects = subject_lengths = pubmed_ids = whitespace_bools = position_ids = question_ids = None

    if len(eval_dataset) == 0:
        return {}, False, eval_loss

    description = "Evaluating" if args.do_predict is False else "Predicting"
    for batch in tqdm(eval_dataloader, desc=description):
        batch_cpu = (batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
        # No need to push everything to GPU
        # question_number = batch[10]
        batch = tuple(batch[i].to(args.device) for i in range(len(batch)) if i <= 4)

        # if question_number < len(lambda_weights):
        #     lambda_weight = torch.FloatTensor([lambda_weights[question_number]] * args.n_gpu)
        # else:
        lambda_weight = torch.FloatTensor([1.0] * args.n_gpu)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels": batch[3],
                      "max_context": batch[4],
                      "lambda_weight": lambda_weight}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.sum()  # sum() to aggregate on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        # Update parameters
        preds, out_label_ids, out_token_ids, is_max_token = update_with_nn_output(
            inputs, logits, preds, out_label_ids, out_token_ids, is_max_token)
        pubmed_ids, subjects, whitespace_bools, position_ids, question_ids, subject_lengths = update_metadata(
            batch_cpu[1], batch_cpu[0], batch_cpu[2], batch_cpu[3], batch_cpu[4], batch_cpu[6], pubmed_ids, subjects,
            whitespace_bools, position_ids, question_ids, subject_lengths, subject_label_encoder, file_names_encoder)

    eval_loss = eval_loss  # / nb_eval_steps
    preds = preds.reshape(out_label_ids.shape[0], out_label_ids.shape[1], preds.shape[2])
    preds = np.argmax(preds, axis=2)

    # index are on token basis, not character
    groundtruth, predictions, out_label_list, preds_list = extract_from_nn_output(
        labels, out_label_ids, preds, out_token_ids, is_max_token, whitespace_bools,
        position_ids, tokenizer, pubmed_ids, subjects, question_ids, subject_lengths)
    # logger.info(out_label_ids.shape)
    # logger.info(len(out_label_list))
    # logger.info(pubmed_ids)

    if args.do_predict is False:
        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
        if reports:
            results["classification_report"] = classification_report(out_label_list, preds_list)
        if not wandb_log:
            logger.info("***** Eval results %s *****", prefix)
            logger.info("  Question %s! ", question_number)
        for key in sorted(results.keys()):
            if args.wandb and wandb_log and args.get_predictions and key == "f1":
                wandb.log({"{}_{}_{}".format(mode, question_number, key): results[key]})
            else:
                logger.info("  %s = %s", key, str(results[key]))

    if args.get_predictions:
        return predictions, True, eval_loss
    else:
        return groundtruth, True, eval_loss


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if args.wandb:
        wandb.init(project="studienprojekt")
        wandb.config.update(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    # logging.basicConfig(filename=root_path + '/output_models/statistics_2.log', format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #                     datefmt="%m/%d/%Y %H:%M:%S",
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare task
    labels = datatypes.EVENT_TAGGING_LABELS
    if args.task == "GE" or args.task == "GE_debug":
        labels = datatypes.EVENT_TAGGING_LABELS2
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = BertConfig, BertForQuestionAnswerTagging, BertTokenizer
    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    if args.wandb:
        wandb.watch(model, log="all")

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(
                args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", data_dir=args.dev_data,
                     prefix=global_step, reports=True)
    if args.wandb:
        wandb.save("model.h1")

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", data_dir=args.test_data,
                     prefix=global_step)


if __name__ == "__main__":
    main()
