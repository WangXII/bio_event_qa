""" Utility functions for the neural network and tensor manipulation of run_mtqa.py """

from __future__ import absolute_import, division, print_function
import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import numpy as np
import torch
import wandb

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch._six import container_abcs
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from metrics.sequence_labeling import get_entities_with_names

logger = logging.getLogger(__name__)


def extract_from_tensor_dataset(dataset, subjects_encoder, file_names_encoder, labels, tokenizer):
    ''' Extract entities from given tensor dataset '''

    dataloader = DataLoader(dataset)
    out_label_ids = out_token_ids = is_max_token = pubmed_ids = subjects = whitespace_bools = position_ids = question_ids = None
    for data_point in dataloader:  # "Batch size" is 1
        if out_label_ids is None:
            out_label_ids = data_point[3].numpy()
            out_token_ids = data_point[0].numpy()
            is_max_token = data_point[4].numpy()
            pubmed_ids = file_names_encoder.inverse_transform(data_point[6].numpy()).tolist()
            subjects = [subjects_encoder.inverse_transform(data_point[5].numpy().ravel()).tolist()[:data_point[11].item()]]
            whitespace_bools = data_point[7].numpy()
            position_ids = data_point[8].numpy()
            question_ids = data_point[9].numpy()
        else:
            out_label_ids = np.append(out_label_ids, data_point[3].numpy(), axis=0)
            out_token_ids = np.append(out_token_ids, data_point[0].numpy(), axis=0)
            is_max_token = np.append(is_max_token, data_point[4].numpy(), axis=0)
            pubmed_ids.extend(file_names_encoder.inverse_transform(data_point[6].numpy()).tolist())
            subjects.append(subjects_encoder.inverse_transform(data_point[5].numpy().ravel()).tolist()[:data_point[11].item()])
            whitespace_bools = np.append(whitespace_bools, data_point[7].numpy(), axis=0)
            position_ids = np.append(position_ids, data_point[8].numpy(), axis=0)
            question_ids = np.append(question_ids, data_point[9].numpy(), axis=0)

    label_map = {i: label for i, label in enumerate(labels)}
    amount_of_questions = max(question_ids) + 1  # question_ids start with 0
    out_label_dict = [[] for i in range(amount_of_questions)]
    token_dict = [[] for i in range(amount_of_questions)]
    whitespace_dict = [[] for i in range(amount_of_questions)]
    position_dict = [[] for i in range(amount_of_questions)]
    pubmed_list = ["" for i in range(amount_of_questions)]
    subject_list = [[] for i in range(amount_of_questions)]

    groundtruth = {}
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if is_max_token[i, j] == 1:
                if out_label_dict[question_ids[i]] == []:
                    pubmed_list[question_ids[i]] = pubmed_ids[i]
                    subject_list[question_ids[i]] = subjects[i]
                out_label_dict[question_ids[i]].append(label_map[out_label_ids[i][j]])
                token_dict[question_ids[i]].append(tokenizer.convert_ids_to_tokens(out_token_ids[i][j].item()))
                whitespace_dict[question_ids[i]].append(whitespace_bools[i][j])
                position_dict[question_ids[i]].append(position_ids[i][j])

    for i in range(len(out_label_dict)):
        answer_list = get_entities_with_names(token_dict[i], out_label_dict[i], whitespace_dict[i], position_dict[i])
        if pubmed_list[i] in groundtruth:
            groundtruth[pubmed_list[i]].append((subject_list[i], answer_list))
        else:
            groundtruth[pubmed_list[i]] = [(subject_list[i], answer_list)]
    return groundtruth


def train_example(args, model, step, batch, optimizer, scheduler, amp, tr_loss, global_step, lambda_weights, iteration):
    """ Train one example in the training loop """
    # question_number = batch[10]
    batch = tuple(batch[i].to(args.device) for i in range(len(batch)) if i <= 4)  # No need to push everything to GPU
    # TODO Add lambda weight support in mixed batches?
    # if question_number < len(lambda_weights):
    #     lambda_weight = torch.FloatTensor([lambda_weights[question_number]] * args.n_gpu)
    # else:
    lambda_weight = torch.FloatTensor([1.0] * args.n_gpu)
    lambda_weight.to(args.device)
    inputs = {"input_ids": batch[0],
              "attention_mask": batch[1],
              "token_type_ids": batch[2],
              "labels": batch[3],
              "max_context": batch[4],
              "lambda_weight": lambda_weight}
    outputs = model(**inputs)
    loss, logits = outputs[:2]  # model outputs are always tuple in pytorch-transformers (see doc)
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    tr_loss += loss.item()
    if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
        if args.wandb:
            wandb.log({"average loss": tr_loss / global_step})

    # logger.info(type(model))
    # batch = tuple(logger.info("Batch {} {}".format(i, batch[i].element_size() * batch[i].nelement()))
    #               for i in range(len(batch)))
    # logger.info("Input_IDs: {}".format(batch[0]))
    # logger.info("ids: {}".format(inputs["input_ids"]))
    # logger.info("Length of ids: {}".format(inputs["input_ids"].shape))
    # logger.info("Lambda: {}".format(lambda_weight.element_size() * lambda_weight.nelement()))
    # logger.info("weight: {}".format(lambda_weight))
    # # logger.info("Inputs {}".format(inputs.element_size() * inputs.nelement()))
    # logger.info("Logits: {}".format(logits.element_size() * logits.nelement()))
    # logger.info("Loss: {}".format(loss.element_size() * loss.nelement()))
    # exit()
    # del lambda_weight
    # del inputs
    # del logits
    # del loss
    # torch.cuda.empty_cache()
    return tr_loss, global_step


def update_with_nn_output(inputs, logits, preds, out_label_ids, out_token_ids, is_max_token):
    if out_label_ids is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        out_token_ids = inputs["input_ids"].detach().cpu().numpy()
        is_max_token = inputs["max_context"].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        out_token_ids = np.append(out_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
        is_max_token = np.append(is_max_token, inputs["max_context"].detach().cpu().numpy(), axis=0)
    return preds, out_label_ids, out_token_ids, is_max_token


def update_metadata(input_pubmed_ids, input_subjects, input_whitespaces, input_positions, input_question_ids, input_subject_lengths,
                    pubmed_ids, subjects, whitespace_bools, position_ids, question_ids, subject_lengths, subject_label_encoder, file_names_encoder):
    if pubmed_ids is None:
        pubmed_ids = file_names_encoder.inverse_transform(input_pubmed_ids.numpy())
        subject_lengths = input_subject_lengths.numpy()
        subjects = subject_label_encoder.inverse_transform(input_subjects.numpy().ravel()).reshape(input_pubmed_ids.numpy().shape[0], -1)
        whitespace_bools = input_whitespaces.numpy()
        position_ids = input_positions.numpy()
        question_ids = input_question_ids.numpy()
    else:
        pubmed_ids = np.append(pubmed_ids, file_names_encoder.inverse_transform(input_pubmed_ids.numpy()), axis=0)
        subject_lengths = np.append(subject_lengths, input_subject_lengths.numpy(), axis=0)
        subjects = np.append(subjects, subject_label_encoder.inverse_transform(input_subjects.numpy().ravel()).reshape(input_pubmed_ids.numpy().shape[0], -1), axis=0)
        whitespace_bools = np.append(whitespace_bools, input_whitespaces.numpy(), axis=0)
        position_ids = np.append(position_ids, input_positions.numpy(), axis=0)
        question_ids = np.append(question_ids, input_question_ids.numpy(), axis=0)
    return pubmed_ids, subjects, whitespace_bools, position_ids, question_ids, subject_lengths


def extract_from_nn_output(labels, out_label_ids, preds, out_token_ids, is_max_token, whitespace_bools, position_ids,
                           tokenizer, pubmed_ids, subjects, question_ids, subject_lengths):
    # Label 'O' has the number 0. We use it for all not discovered protein theme event pairs
    label_map = {i: label for i, label in enumerate(labels)}
    # out_labels_id[0]: number of questions (943 in our eval set)
    # out_labels_id[1]: combined sequence length (n times maximum sequence length with padding)
    # out_label_list = [[] for _ in range(max(question_ids))]
    # preds_list = [[] for _ in range(max(question_ids))]
    # token_list = [[] for _ in range(max(question_ids))]
    # whitespace_list = [[] for _ in range(max(question_ids))]

    amount_of_questions = max(question_ids) + 1  # question_ids start with 0
    out_label_dict = [[] for i in range(amount_of_questions)]
    preds_dict = [[] for i in range(amount_of_questions)]
    token_dict = [[] for i in range(amount_of_questions)]
    whitespace_dict = [[] for i in range(amount_of_questions)]
    position_dict = [[] for i in range(amount_of_questions)]
    pubmed_list = ["" for i in range(amount_of_questions)]
    subject_list = [[] for i in range(amount_of_questions)]

    # np.set_printoptions(threshold=sys.maxsize)
    # logger.info(max(question_ids))
    # logger.info(len(question_ids))
    # logger.info(question_ids)
    # logger.info(out_label_ids.shape)
    groundtruth = {}
    predictions = {}
    # logger.info("HALLO")
    # logger.info(len(out_label_dict))
    # logger.info("FOo")
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            # if out_label_ids[i, j] != pad_token_label_id:
            if is_max_token[i, j] == 1:
                # logger.info(question_ids[i])
                if out_label_dict[question_ids[i]] == []:
                    pubmed_list[question_ids[i]] = pubmed_ids[i]
                    subject_list[question_ids[i]] = subjects[i][:subject_lengths[i]]
                out_label_dict[question_ids[i]].append(label_map[out_label_ids[i][j]])
                preds_dict[question_ids[i]].append(label_map[preds[i][j]])
                token_dict[question_ids[i]].append(tokenizer.convert_ids_to_tokens(out_token_ids[i][j].item()))
                whitespace_dict[question_ids[i]].append(whitespace_bools[i][j])
                position_dict[question_ids[i]].append(position_ids[i][j])

    # logger.info(pubmed_ids)
    # logger.info(pubmed_list)
    # Python dicts retain order automatically since version 3.6
    # out_label_list = list(out_label_dict.values())
    # preds_list = list(preds_dict.values())
    # logger.info(out_label_list)
    # logger.info("Number of Questions:")
    # logger.info(len(out_label_dict))
    for i in range(len(out_label_dict)):
        answer_list = get_entities_with_names(token_dict[i], out_label_dict[i], whitespace_dict[i], position_dict[i])
        answer_list2 = get_entities_with_names(token_dict[i], preds_dict[i], whitespace_dict[i], position_dict[i])
        if pubmed_list[i] in groundtruth:
            groundtruth[pubmed_list[i]].append((subject_list[i], answer_list))
        else:
            groundtruth[pubmed_list[i]] = [(subject_list[i], answer_list)]
        if pubmed_list[i] in predictions:
            # predictions[pubmed_ids[i]].append(([subjects[i]], answer_list))  # Build upon groundtruth, check best case
            predictions[pubmed_list[i]].append((subject_list[i], answer_list2))
        else:
            # predictions[pubmed_ids[i]] = [([subjects[i]], answer_list)]  # Build upon groundtruth, check best case
            predictions[pubmed_list[i]] = [(subject_list[i], answer_list2)]

    # logger.info("Number of PubMed IDs")
    # logger.info(len(groundtruth))
    # logger.info(groundtruth)
    # logger.info(groundtruth["PMID-10369669"])
    # logger.info(predictions["PMID-12824002"])
    # exit()
    return groundtruth, predictions, out_label_dict, preds_dict


def initialize_loaders(args, dataset, optimizer):
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
    total = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps, num_training_steps=total)
    return dataloader, scheduler


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if batch[0].nelement() == 1:
            out = None
            return torch.stack(batch, 0, out=out)
        else:
            return pad_sequence(batch, True)
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError("collate_fn: batch must contain tensors or lists; found {}".format(elem_type))
