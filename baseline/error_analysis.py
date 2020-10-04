'''Analyse errors'''

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import pickle
import re
from data_processing.file_converter import getEventAnnotations, getProteinAnnotations
from metrics.tuples import debugInformation

logger = logging.getLogger(__name__)


def strip_non_letters(string_word):
    return re.sub("[^a-zA-Z_]+", "", string_word)


def convert_a2File(directory, file):
    """ Converts the .a2 file in given directory according to custom eval format
    Parameters
    ----------
    directory : str
        The directory location as string
    file : str
        The file location as string

    Returns
    -------
    list
        list of event sets all for six question
    """

    pubmed_id = file[:-4]
    proteins = getProteinAnnotations(directory + "/" + file[:-4] + ".a1")
    triggers, events, equivs = getEventAnnotations(directory + "/" + file[:-4] + ".a2")
    proteins_and_triggers = {**proteins, **triggers}
    event_list = [set(), set(), set(), set(), set(), set()]

    # First question
    for event in events.values():
        event_trigger = triggers[event.arguments[event.name]].name
        event_label = event.name
        if "Theme" in event.arguments and event.arguments["Theme"].startswith("T"):
            for theme_key, theme_id in event.arguments.items():
                if theme_key.startswith("Theme"):
                    theme = proteins_and_triggers[event.arguments[theme_key]].name
                    event_list[0].add((event_label, (event_trigger, strip_non_letters(event_label), theme, "Theme", pubmed_id)))
        elif "Participant" in event.arguments and event.arguments["Participant"].startswith("T"):
            for theme_key, theme_id in event.arguments.items():
                if theme_key.startswith("Participant"):
                    theme = proteins_and_triggers[event.arguments[theme_key]].name
                    event_list[0].add((event_label, (event_trigger, strip_non_letters(event_label), theme, "Theme", pubmed_id)))
            # print((event_label, (event_trigger, event_label, theme, pubmed_id)))
            # exit()

    # Second question
    for event in events.values():
        event_trigger = triggers[event.arguments[event.name]].name
        event_label = event.name
        if "Theme" in event.arguments and event.arguments["Theme"].startswith("T"):
            for argument_key, argument_id in event.arguments.items():
                if argument_key != event.name:
                    if event.arguments[argument_key].startswith("T"):
                        argument_name = proteins_and_triggers[event.arguments[argument_key]].name
                    else:
                        event_argument = events[event.arguments[argument_key]]
                        argument_name = triggers[event_argument.arguments[event_argument.name]].name
                    for theme_key, theme_id in event.arguments.items():
                        if theme_key.startswith("Theme") and argument_key != theme_key:
                            theme = proteins_and_triggers[event.arguments[theme_key]].name
                            event_list[1].add((strip_non_letters(argument_key), (argument_name, strip_non_letters(argument_key), event_label.lower(),
                                               strip_non_letters(event_label), theme, "Theme", pubmed_id)))
        elif "Participant" in event.arguments and event.arguments["Participant"].startswith("T"):
            for argument_key, argument_id in event.arguments.items():
                if argument_key != event.name:
                    if event.arguments[argument_key].startswith("T"):
                        argument_name = proteins_and_triggers[event.arguments[argument_key]].name
                    else:
                        event_argument = events[event.arguments[argument_key]]
                        argument_name = triggers[event_argument.arguments[event_argument.name]].name
                    for theme_key, theme_id in event.arguments.items():
                        if theme_key.startswith("Participant") and argument_key != theme_key:
                            theme = proteins_and_triggers[event.arguments[theme_key]].name
                            event_list[1].add((strip_non_letters(argument_key), (argument_name, "Participant", event_label.lower(),
                                               strip_non_letters(event_label), theme, "Theme", pubmed_id)))

    # Third question
    for regulation in events.values():
        regulation_trigger = triggers[regulation.arguments[regulation.name]].name
        regulation_label = regulation.name
        if "Theme" in regulation.arguments and regulation.arguments["Theme"].startswith("E"):
            event = events[regulation.arguments["Theme"]]
            event_trigger = triggers[event.arguments[event.name]].name
            event_label = event.name
            if "Theme" in event.arguments and event.arguments["Theme"].startswith("T"):
                for theme_key, theme_id in event.arguments.items():
                    if theme_key.startswith("Theme"):
                        theme = proteins_and_triggers[event.arguments[theme_key]].name
                        event_list[2].add((regulation_label, (regulation_trigger, regulation_label, event_label.lower(),
                                           strip_non_letters(event_label), theme, "Theme", pubmed_id)))

    # Forth question
    for regulation in events.values():
        regulation_trigger = triggers[regulation.arguments[regulation.name]].name
        regulation_label = regulation.name
        if "Theme" in regulation.arguments and regulation.arguments["Theme"].startswith("E"):
            event = events[regulation.arguments["Theme"]]
            event_trigger = triggers[event.arguments[event.name]].name
            event_label = event.name
            if "Theme" in event.arguments and event.arguments["Theme"].startswith("T"):
                for argument in regulation.arguments:
                    if argument != "Theme" and argument != regulation_label:
                        if regulation.arguments[argument].startswith("T"):
                            argument_name = proteins_and_triggers[regulation.arguments[argument]].name
                        else:
                            reg_argument = events[regulation.arguments[argument]]
                            argument_name = triggers[reg_argument.arguments[reg_argument.name]].name
                        for theme_key, theme_id in event.arguments.items():
                            if theme_key.startswith("Theme"):
                                theme = proteins_and_triggers[event.arguments[theme_key]].name
                                event_list[3].add((argument, (argument_name, argument, regulation_label.lower(),
                                                   regulation_label, event_label.lower(), strip_non_letters(event_label), theme, "Theme", pubmed_id)))

    # Fifth question
    for reg2 in events.values():
        reg2_trigger = triggers[reg2.arguments[reg2.name]].name
        reg2_label = reg2.name
        if "Theme" in reg2.arguments and reg2.arguments["Theme"].startswith("E"):
            regulation = events[reg2.arguments["Theme"]]
            regulation_trigger = triggers[regulation.arguments[regulation.name]].name
            regulation_label = regulation.name
            if "Theme" in regulation.arguments and regulation.arguments["Theme"].startswith("E"):
                event = events[regulation.arguments["Theme"]]
                event_trigger = triggers[event.arguments[event.name]].name
                event_label = event.name
                if "Theme" in event.arguments and event.arguments["Theme"].startswith("T"):
                    for theme_key, theme_id in event.arguments.items():
                        if theme_key.startswith("Theme"):
                            theme = proteins_and_triggers[event.arguments[theme_key]].name
                            event_list[4].add((reg2_label, (reg2_trigger, reg2_label, regulation_label.lower(), regulation_label,
                                               event_label.lower(), strip_non_letters(event_label), theme, "Theme", pubmed_id)))

    # Sixth
    for reg2 in events.values():
        reg2_trigger = triggers[reg2.arguments[reg2.name]].name
        reg2_label = reg2.name
        if "Theme" in reg2.arguments and reg2.arguments["Theme"].startswith("E"):
            regulation = events[reg2.arguments["Theme"]]
            regulation_trigger = triggers[regulation.arguments[regulation.name]].name
            regulation_label = regulation.name
            if "Theme" in regulation.arguments and regulation.arguments["Theme"].startswith("E"):
                event = events[regulation.arguments["Theme"]]
                event_trigger = triggers[event.arguments[event.name]].name
                event_label = event.name
                if "Theme" in event.arguments and event.arguments["Theme"].startswith("T"):
                    for argument in reg2.arguments:
                        if argument != "Theme" and argument != reg2_label:
                            if reg2.arguments[argument].startswith("T"):
                                argument_name = proteins_and_triggers[reg2.arguments[argument]].name
                            else:
                                reg_argument = events[reg2.arguments[argument]]
                                argument_name = triggers[reg_argument.arguments[reg_argument.name]].name
                            for theme_key, theme_id in event.arguments.items():
                                if theme_key.startswith("Theme"):
                                    theme = proteins_and_triggers[event.arguments[theme_key]].name
                                    event_list[5].add((argument, (argument_name, argument, reg2_label.lower(), reg2_label,
                                                       regulation_label.lower(), regulation_label, event_label.lower(), strip_non_letters(event_label),
                                                       theme, "Theme", pubmed_id)))

    return event_list


def convertDirectory(directory):
    """ Converts all .a2 files in given directory according to custom eval format
    Parameters
    ----------
    directory : str
        The directory location as string

    Returns
    -------
    list
        list of event sets all for six questions
    """

    event_list = [set(), set(), set(), set(), set(), set()]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            result = convert_a2File(directory, filename)
            event_list = [event_list[i] | result[i] for i in range(6)]
    return event_list


def convertOutput(nn_output):
    """ Converts all outputs from NN to custom eval format
    Parameters
    ----------
    list :
        List of outputs from NN
        [ {pubmed_id : [([subjects], [(entity, label, start_index, end_index)])]} ]

    Returns
    -------
    list
        list of event sets all for six questions
    """

    event_list = [set(), set(), set(), set(), set(), set()]
    # logger.info(nn_output)
    for i, number in enumerate(nn_output):
        print(i)
        if i >= 6:
            continue
        for pubmed_id, answers in number.items():
            # pubmed_string = "PMID-" + str(pubmed_id)
            for subjects_entity in answers:
                subjects = [x.lower() if i % 2 == 0 else x for i, x in enumerate(subjects_entity[0])]
                for entity in subjects_entity[1]:
                    result = [x.lower() if i % 2 == 0 else x for i, x in enumerate(entity[:-4])]
                    event_list[i].add((entity[1], tuple(result + subjects + [pubmed_id])))
                    # logger.info((entity[1], tuple(list(entity[:-2]) + subjects + [pubmed_string])))
                    # exit()
    return event_list


def evaluateAllQuestions(groundtruth_list, predictions_list):
    """ Prints classification report for all six questions
    Parameters
    ----------
    groundtruth_list :
        list of event sets all for six question
    predictions_list :
        list of event sets all for six question
    debug :
        bool. Whether or not to print false positives and false negatives
    """

    debugInformation(groundtruth_list, predictions_list)


if __name__ == "__main__":
    # groundtruth_directory = root_path + "/data/data/PathwayCuration/dev"
    # baseline_directory = root_path + "/working_directory_TEES/output_dev/events"

    # Pathway Curation
    # groundtruth_directory = root_path + "/data/data/PathwayCuration/dev"
    groundtruth_directory = root_path + "/data/data/PathwayCuration/BioNLP-ST_2013_PC_development_data"
    # baseline_directory = root_path + "/working_directory_TEES/output_test/events"
    # baseline_directory = root_path + "/working_directory_TEES/model_dev2/classification-test/events"
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/PC13_dev"
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/PC13_dev_ensemble"
    # baseline_directory = root_path + "/working_directory_TEES/model_dev_pc/classification-test/events"

    # GENIA
    groundtruth_directory = root_path + "/data/data/Genia11/BioNLP-ST_2011_genia_devel_data_rev1"
    # baseline_directory = root_path + "/working_directory_TEES/model_dev_genia/classification-test/events"
    # GENIA cnn
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/GE11_dev"
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/GE11_dev_ensemble"

    truth = set([("Phosphorylation", ("Phosphorylation", "phosphorylates", "ACE1", 19556))])
    pred = set([("Phosphorylation", ("Phosphorylation", "phosphorylates", "ACE1", 19556))])
    # print(getClassificationReport(truth, pred))

    # print(convert_a2File("/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/"
    #                      + "data/data/PathwayCuration/train", "PMID-1437141.txt"))

    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    groundtruth = convertDirectory(groundtruth_directory)
    # baseline = convertDirectory(baseline_directory, True)
    with open(root_path + "/output_files/event_tuples_GE_dev.npy", 'rb') as f:
        results = pickle.load(f)
        model = convertOutput(results[1:])
        # print(groundtruth)
        # print(model[:1])
        evaluateAllQuestions(groundtruth, model)
