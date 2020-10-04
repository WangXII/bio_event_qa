# question_tagger.py
# Xing Wang
# Created: 25/02/2020
# Last changed: 08/03/2020

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging

logger = logging.getLogger(__name__)


def tagQuestion(question, context, answers, subject=[], pubmed_id=-1):
    """ Tags all event arguments
    Parameters
    ----------
    question : list of str
        the question denoted as a list of strings
    context : list
        given text as a list of Tuples (token, start_index, end_index)
    answers : list
        List of (argument_type, protein_or_event_trigger) or (regulation_event_type, event_trigger)
    subject: list
        List of proteins/event triggers we are asking for
    pubmed_id: long
        PubMed ID of the corresponding question

    Returns
    -------
    list
        The annotated events by BERT in IOB format (token), whether a whitespace is after,
        and start and end positions of the tokens. Checking for maximum sequence length not done here
    """

    tagged_question = [pubmed_id, subject]
    tagged_question.append(("[CLS]", 'O', False, -1, -1))
    for token in question:
        tagged_question.append((token, 'O', False, -1, -1))
    tagged_question.append(("[SEP]", 'O', False, -1, -1))
    t = 0  # Current Event Index
    state = 0  # 0 for 'O' and 'I', 1 for 'B'
    for w, word in enumerate(context):
        whitespace = False  # no whitespace before the token
        if (w >= 1) and word[1] > context[w - 1][2] + 1:
            whitespace = True
        if word[0].startswith("##"):
            tagged_question.append((word[0], 'X', whitespace, word[1], word[2]))
        elif len(answers) == 0:
            # no events for the protein, all tokens 'O'
            tagged_question.append((word[0], 'O', whitespace, word[1], word[2]))
        elif (word[1] >= int(answers[t][1].start_index)) \
                and (word[2] < int(answers[t][1].end_index)) \
                and (state == 1):
            tagged_question.append((word[0], 'I-' + answers[t][0], whitespace, word[1], word[2]))
        elif (word[1] >= int(answers[t][1].start_index)) \
                and (word[2] < int(answers[t][1].end_index)):
            tagged_question.append((word[0], 'B-' + answers[t][0], whitespace, word[1], word[2]))
            state = 1
        else:
            tagged_question.append((word[0], 'O', whitespace, word[1], word[2]))
            state = 0
        if len(answers) != 0 \
                and (int(answers[t][1].end_index) <= word[2] + 1) \
                and (t < len(answers) - 1):
            t += 1
            state = 0
    tagged_question.append(("[SEP]", 'O', whitespace, -1, -1))
    return tagged_question
