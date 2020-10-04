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

""" Helps creating question answering examples. """

from __future__ import absolute_import, division, print_function

import logging
from sklearn import preprocessing

import data_processing.file_converter as file_converter

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, whitespaces, positions, question_end,
                 subjects=[], pubmed_id=-1, question_id=-1):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: list. The labels for each word of the sequence. This should be
                specified for train and dev examples, but not for test examples.
            whitespaces: list. List for each token the bool whether a whitespace follow afterwards.
            positions: list of tuples. Tuple with start and end position of each token.
            question_end: int. The index where the question ends and the text span begins.
            subjects: list. Subjects of interest encoded in ints.
            pubmed_id: String. PubMed ID.
            question_id: list. Single unique ID for all questions passed in the model.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.whitespaces = whitespaces
        self.positions = positions
        self.question_end = question_end
        self.subjects = subjects
        self.pubmed_id = pubmed_id
        self.question_id = question_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, whitespace_bools, position_ids,
                 token_is_max_context, subjects=[], pubmed_id=-1, question_id=-1):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.whitespace_bools = whitespace_bools
        self.position_ids = position_ids
        self.token_is_max_context = token_is_max_context
        self.subjects = subjects
        self.pubmed_id = pubmed_id
        self.question_id = question_id
        self.subject_length = len(subjects)


def generate_examples(directory_loc, mode, recursion_depth, question_type, prior_events=None, predict=False):
    guid_index = 1
    examples = []
    files, new_answers = file_converter.generateAnnotationsFromDirectory(directory_loc, recursion_depth, question_type, prior_events, predict)
    all_subjects = []
    all_file_names = []
    question_id = -1
    for file in files:
        # logger.info(file)
        bool_question_end = False
        question_end = -1
        words = []
        labels = []
        whitespaces = []
        positions = []
        pubmed_id = file[0]
        question_id = question_id + 1
        subjects = file[1]
        all_subjects.extend(subjects)
        all_file_names.append(pubmed_id)
        file = file[2:]
        for i, line in enumerate(file):
            if line[0] == "[SEP]" and not bool_question_end:
                bool_question_end = True
                question_end = i + 1
            words.append(line[0])
            labels.append(line[1])
            whitespaces.append(line[2])
            positions.append([line[3], line[4]])
        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels,
                                     whitespaces=whitespaces, positions=positions, question_end=question_end,
                                     subjects=subjects, pubmed_id=pubmed_id, question_id=question_id))
        assert question_end != -1
    return examples, all_subjects, all_file_names, new_answers


def convert_examples_to_features(examples, label_list, max_seq_length, doc_stride, tokenizer, pad_token=0,
                                 pad_token_segment_id=0, pad_token_label_id=-1, sequence_a_segment_id=0,
                                 sequence_b_segment_id=1, mask_padding_with_zero=True, subject_list=[],
                                 file_names=[]):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    subject_label_encoder = preprocessing.LabelEncoder()
    subject_label_encoder.fit(subject_list)
    file_names_encoder = preprocessing.LabelEncoder()
    file_names_encoder.fit(file_names)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        labels = []
        whitespaces = []
        positions = []
        for word_tokens, label, whitespace, position in zip(example.words, example.labels,
                                                            example.whitespaces, example.positions):
            tokens.append(word_tokens)
            labels.append(label_map[label])
            whitespaces.append(whitespace)
            positions.append(position)

        question_end = example.question_end
        tokens_question = tokens[:question_end]
        tokens_document = tokens[question_end:]
        labels_question = labels[:question_end]
        labels_document = labels[question_end:]
        whitespaces_question = whitespaces[:question_end]
        whitespaces_document = whitespaces[question_end:]
        positions_question = positions[:question_end]
        positions_document = positions[question_end:]

        max_document_length = max_seq_length - len(tokens_question)
        assert max_document_length > 0
        feature_length = max_document_length if len(tokens_document) > max_document_length else len(tokens_document)
        current_token_document = tokens_document[:feature_length]
        remaining_token_document = tokens_document[feature_length:]
        current_label_document = labels_document[:feature_length]
        remaining_label_document = labels_document[feature_length:]
        current_whitespace_document = whitespaces_document[:feature_length]
        remaining_whitespace_document = whitespaces_document[feature_length:]
        current_position_document = positions_document[:feature_length]
        remaining_position_document = positions_document[feature_length:]
        finished = False
        started = False

        while not finished:
            current_tokens = tokens_question + current_token_document
            input_ids = tokenizer.convert_tokens_to_ids(current_tokens)
            segment_ids = [sequence_a_segment_id] * example.question_end + [sequence_b_segment_id] * feature_length
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            label_ids = labels_question + current_label_document
            whitespace_bools = whitespaces_question + current_whitespace_document
            position_ids = positions_question + current_position_document
            token_is_max_context = [0] * max_seq_length
            if not started:
                started = True
                token_is_max_context[question_end: question_end + feature_length] = [1] * feature_length
            else:
                token_is_max_context[question_end + int(doc_stride / 2): question_end + feature_length] =\
                    [1] * (feature_length - int(doc_stride / 2))
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids += ([pad_token] * padding_length)
                input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids += ([pad_token_segment_id] * padding_length)
                label_ids += ([pad_token_label_id] * padding_length)
                whitespace_bools += ([pad_token_label_id] * padding_length)
                position_ids += ([[pad_token_label_id, pad_token_label_id]] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(whitespace_bools) == max_seq_length
            assert len(position_ids) == max_seq_length

            # Update next feature length
            if len(remaining_token_document) > 0:
                token_is_max_context[question_end + feature_length - int(doc_stride / 2):
                                     question_end + feature_length] = [0] * int(doc_stride / 2)
                feature_length = max_document_length\
                    if len(remaining_token_document) + doc_stride > max_document_length\
                    else len(remaining_token_document) + doc_stride
                current_token_document = current_token_document[len(current_token_document) - doc_stride:]\
                    + remaining_token_document[:feature_length - doc_stride]
                remaining_token_document = remaining_token_document[feature_length - doc_stride:]
                current_label_document = current_label_document[len(current_label_document) - doc_stride:]\
                    + remaining_label_document[:feature_length - doc_stride]
                remaining_label_document = remaining_label_document[feature_length - doc_stride:]
                current_whitespace_document = \
                    current_whitespace_document[len(current_whitespace_document) - doc_stride:]\
                    + remaining_whitespace_document[:feature_length - doc_stride]
                remaining_whitespace_document = remaining_whitespace_document[feature_length - doc_stride:]
                current_position_document = \
                    current_position_document[len(current_position_document) - doc_stride:]\
                    + remaining_position_document[:feature_length - doc_stride]
                remaining_position_document = remaining_position_document[feature_length - doc_stride:]
            else:
                finished = True

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              whitespace_bools=whitespace_bools,
                              position_ids=position_ids,
                              token_is_max_context=token_is_max_context,
                              subjects=subject_label_encoder.transform(example.subjects),
                              pubmed_id=file_names_encoder.transform([example.pubmed_id])[0],
                              question_id=example.question_id))

    return features, subject_label_encoder, file_names_encoder
