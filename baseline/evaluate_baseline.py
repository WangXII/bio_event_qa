# file_converter.py
# Xing Wang
# Created: 05.01.2020
# Last changed: 05.01.2020

# Evaluates the baseline performance

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from data_processing.file_converter import *
from metrics.sequence_labeling import precision_score, recall_score, f1_score, classification_report, confusion_matrix_report
from datastructures.datatypes import EVENT_TAGS


def convertToIOBFormat(file_loc, tokenizer=BertTokenizer.from_pretrained(
                       root_path + "/scibert/scibert_scivocab_uncased", do_lower_case=True)):
    """ Create question/answer pairs for all proteins in the file
    Parameters
    ----------
    file_loc : str
        The file location as string
    tokenizer
        SciBert Tokenizer as default

    Returns
    -------
    list
        The annotated events in IOB format (token, tag) and BERT-tokenization
    """

    context = getWordTokenizationPosition(file_loc, tokenizer, {}, {}, -1)
    proteins = getProteinAnnotations(file_loc[:-4] + ".a1")
    triggers, events, equivs = getEventAnnotations(file_loc[:-4] + ".a2")
    protein_themeOf_event_pairs = extractProteinThemeOfEventPairs(
        proteins, triggers, events, equivs)
    # Change protein list to dict
    protein_themeOf_events = {}
    for id, protein in proteins.items():
        protein_themeOf_events[protein.name] = []
    for pair in protein_themeOf_event_pairs:
        protein_themeOf_events[pair[0].name].append(pair[1])
    # traverse all proteins
    result = []
    for protein, event_list in protein_themeOf_events.items():
        question = tokenizer.tokenize("Which events is " + protein
                                      + " a theme of ?")
        answers = event_list
        result.append(tagQuestion(question, context, answers)[2:])
    return result


def convertDirectory(directory_loc):
    """ Create question/answer pairs for all files in the directory
    Parameters
    ----------
    directory_loc : str
        The file location as string

    Returns
    -------
    list
        The annotated events in IOB format (token, tag) and BERT-tokenization
        of all documents in the given directory
    """

    result = []
    for file in sorted(os.listdir(directory_loc)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            result = result + convertToIOBFormat(directory_loc + "/" + filename)
    return result


if __name__ == "__main__":
    groundtruth_directory = root_path + "/data/data/PathwayCuration/dev"
    baseline_directory = root_path + "/TEES/working_directory/output/predictions2/events"

    groundtruth = [list(list(zip(*doc))[1]) for doc in convertDirectory(groundtruth_directory)]
    baseline = [list(list(zip(*doc))[1]) for doc in convertDirectory(baseline_directory)]
    # print(baseline[1])
    # print(groundtruth[1])
    # print(baseline[2])
    # print(groundtruth[2])
    # print(baseline[3])
    # print(groundtruth[3])
    print("Precision: {}".format(precision_score(groundtruth, baseline)))
    print("Recall: {}".format(recall_score(groundtruth, baseline)))
    print("F1-score: {}".format(f1_score(groundtruth, baseline)))
    print(classification_report(groundtruth, baseline))
    print(confusion_matrix_report(groundtruth, baseline, EVENT_TAGS))

