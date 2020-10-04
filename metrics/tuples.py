# Adapted from tensorflow seqeval.metrics.sequence_labelling.py
""" Tuple evaluation """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


def computeMetrics(groundtruth, prediction):
    """Compute the precision, recall and F1 score.

    Args:
        groundtruth : set. Ground truth (correct) target values.
        prediction : set. Estimated targets.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> groundtruth = set([("Phosphorylation", ("Phosphorylation", "phosphorylates", "ACE1"))])
        >>> prediction = set([("Phosphorylation", ("Phosphorylation", "phosphorylates", "ACE1"))])
        >>> f1_score(groundtruth, prediction)
        1.00
    """

    nb_correct = len(groundtruth & prediction)
    nb_pred = len(prediction)
    nb_true = len(groundtruth)

    p = nb_correct / nb_pred * 100 if nb_pred > 0 else 0
    r = nb_correct / nb_true * 100 if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, score


def debugInformation(groundtruth_list, predictions_list):
    """ Log debugging/error analysis information for all six questions
    Parameters
    ----------
    groundtruth_list :
        list of event sets all for six question
    predictions_list :
        list of event sets all for six question
    """

    # logger.info(groundtruth_list)
    # logger.info(predictions_list)

    for i, (groundtruth, prediction) in enumerate(zip(groundtruth_list, predictions_list)):
        print("Question number: " + str(i))

        true_positive = groundtruth & prediction
        print("True positives: ")
        print(len(true_positive))
        # if i == 1:
        #     print(true_positive)

        false_positive = prediction - (groundtruth & prediction)
        print("False positives: ")
        print(len(false_positive))
        # if i == 4:
        #     print(false_positive)

        false_negative = groundtruth - (groundtruth & prediction)
        print("False negatives: ")
        print(len(false_negative))
        # if i == 0:
        #     print(false_negative)

        # Detailed error analysis
        if i != 0:
            predictions_prior = set([example[1][2:] for example in prediction])
            cumulated_error = 0
            for example in groundtruth:
                if example[1][2:] not in predictions_prior:
                    cumulated_error += 1
            print("Cumulated Errors: ")
            print(cumulated_error)

        fp_no_labels = [tuple(list(example[1][0]) + list(example[1][2:])) for example in false_positive]
        fn_no_labels = [tuple(list(example[1][0]) + list(example[1][2:])) for example in false_negative]
        wrong_labels_1 = set(fn_no_labels) & set(fp_no_labels)
        count_wrong_labels_1 = 0
        for wrong_label in wrong_labels_1:
            count_wrong_labels_1 += Counter(fp_no_labels)[wrong_label]
        print("Wrong Labels: ")
        # if i == 1:
        #     print(wrong_labels_1)
        print(count_wrong_labels_1)

        fn_no_labels = [tuple(list(example[1][0]) + list(example[1][2:])) for example in groundtruth]
        wrong_labels_1 = set(fn_no_labels) & set(fp_no_labels)
        count_wrong_labels_1 = 0
        for wrong_label in wrong_labels_1:
            count_wrong_labels_1 += Counter(fp_no_labels)[wrong_label]
        # print("Wrong Labels All: ")
        # logger.info(wrong_labels_1)
        # print(count_wrong_labels_1)

        entities = set([entity[1][0] for entity in groundtruth])
        count_wrong_span = 0
        for example in false_positive:
            list_substrings = []
            for entity in entities:
                if ((example[1][0] in entity) or (entity in example[1][0])) and (entity != example[1][0]):
                    list_substrings.append(example[1][0])
            for substring in list_substrings:
                event = (example[0], tuple([substring] + list(example[1][1:])))
                if event in groundtruth:
                    count_wrong_span += 1
                    continue
        print("Right label, but entity A is a substring or superstring of entity B")
        print(count_wrong_span)

        print("")


def getClassificationReport(groundtruth, prediction, digits=2):
    """Build a text report showing the main classification metrics.

    Args:
        groundtruth : set. Ground truth (correct) target values.
        prediction : set. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.
    """

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in groundtruth:
        d1[e[0]].add(e[1])
        name_width = max(name_width, len(e[0]))
    for e in prediction:
        d2[e[0]].add(e[1])

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name in sorted(d1):
        true_entities = d1[type_name]
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred * 100 if nb_pred > 0 else 0
        r = nb_correct / nb_true * 100 if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    # report += u'\n'

    # # compute averages
    # report += row_fmt.format('micro avg',
    #                          *computeMetrics(groundtruth, prediction),
    #                          np.sum(s),
    #                          width=width, digits=digits)
    # if s != []:
    #     report += row_fmt.format('last_line_heading',
    #                              np.average(ps, weights=s),
    #                              np.average(rs, weights=s),
    #                              np.average(f1s, weights=s),
    #                              np.sum(s),
    #                              width=width, digits=digits)

    return report, computeAverages(groundtruth, prediction, ps, rs, f1s, s, width, digits)


def computeAverages(groundtruth, prediction, ps, rs, f1s, s, width=9, digits=2):
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    report = u''
    report += row_fmt.format('micro avg',
                             *computeMetrics(groundtruth, prediction),
                             np.sum(s),
                             width=width, digits=digits)
    if s != []:
        report += row_fmt.format('macro avg',
                                 np.average(ps, weights=s),
                                 np.average(rs, weights=s),
                                 np.average(f1s, weights=s),
                                 np.sum(s),
                                 width=width, digits=digits)

    return report, np.average(ps, weights=s), np.average(rs, weights=s), np.average(f1s, weights=s), np.sum(s)


if __name__ == "__main__":
    truth = set([("Phosphorylation", ("phosphorylates", "Phosphorylation", "ACE1", 19956))])
    pred = set([("Phosphorylation", ("phosphorylates", "Phosphorylation", "ACE1", 19956))])
    print(truth & pred)
    precision, recall, f1 = computeMetrics(truth, pred)
    print(f1)
    print(getClassificationReport(truth, pred))
