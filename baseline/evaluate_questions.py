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

import logging
import pickle

from data_processing.file_converter import getEventAnnotations, getProteinAnnotations
from metrics.tuples import getClassificationReport, computeAverages, computeMetrics
from datastructures.datatypes import REGULATION_TYPES, CONVERSION_TYPES, EVENT_TAGS

logger = logging.getLogger(__name__)


def convert_QATuples(event_tuples, protein_directory, doc_level=True):
    """ Converts the given event tuples to knowledge base tuples. Evaluate non-merging questions
    Parameters
    ----------
    event_tuples : list
        List of outputs from NN
        [ {pubmed_id : [([subjects], [(entity, label, start_index, end_index)])]} ]

    Returns
    -------
    list
        list of event sets all for six questions
    """

    event_list = [set(), set(), set(), set()]
    # logger.info(nn_output)
    for i, number in enumerate(event_tuples):
        for pubmed_id, answers in number.items():
            # pubmed_string = "PMID-" + str(pubmed_id)
            for subjects_entity in answers:
                subjects = []
                for x_number, x in enumerate(subjects_entity[0]):
                    if (x_number % 2 == 0) and (subjects_entity[0][x_number + 1] in EVENT_TAGS):
                        subjects.append(subjects_entity[0][x_number + 1])
                    elif (x_number % 2 == 0) and (subjects_entity[0][x_number + 1] not in EVENT_TAGS):
                        if (x_number + 1 < len(subjects_entity[0])) - 1:
                            subjects.append(subjects_entity[0][x_number + 1])
                        subjects.append(x.lower())
                for entity in subjects_entity[1]:
                    result = [x.lower() if i % 2 == 0 else x for i, x in enumerate(entity[:-4])]
                    if i == 0:  # Simple Events
                        event_list[0].add((entity[1], tuple(result[1:] + subjects + [pubmed_id])))
                    elif i == 1:  # Simple Events
                        event_list[1].add((subjects[0] + "_" + entity[1], tuple(result[::-1] + subjects + [pubmed_id])))
                    if i % 2 == 0 and i >= 2:  # Nested Events
                        event_list[2].add(("REGULATION", tuple(["REGULATION"] + subjects[-2:] + [pubmed_id])))
                    elif i % 2 == 1 and i >= 2:  # Nested Events
                        event_list[3].add(("Cause", tuple(result[::-1] + ["REGULATION"] + subjects[-2:] + [pubmed_id])))
    # print(event_list)
    return event_list


def convert_a2File2(directory, filename, genia=False, doc_level=True):
    """ Converts the .a2 file in given directory according to questions of a knowledge base evaluation.

    This converting is done with the question answering approach in mind so that no merging is neeeded

    Parameters
    ----------
    directory : str
        The directory location as string
    filename : str
        The .a2 file name, e.g., PMID-1234.a2
    doc_level : bool
        True is document level, False is corpus level evaluation

    Returns
    -------
    list
        list of event tuples for all questions
    """

    pubmed_id = filename[:-3]
    proteins = getProteinAnnotations(directory + "/" + filename[:-3] + ".a1")
    triggers, events, equivs = getEventAnnotations(directory + "/" + filename)
    entities = {**proteins, **triggers}
    # Four question Types
    event_list = [set() for i in range(4)]

    # Question Type 1: Theme/Participant Event Pairs, almost the same as with the merging approach below
    for event in events.values():
        event_label = event.name
        for argument_key, argument in event.arguments.items():
            if argument_key.startswith(("Theme", "Participant")) and argument.startswith("T"):
                theme = proteins[argument].name
                if doc_level is True:
                    event_list[0].add((event_label, (event_label, theme, pubmed_id)))
                else:
                    event_list[0].add((event_label, (event_label, theme)))

    # Question Type 2: One Argument for the event
    for event in events.values():
        event_label = event.name
        for theme_label, theme_id in event.arguments.items():
            if theme_label.startswith(("Theme", "Participant")) and theme_id.startswith("T"):
                theme = proteins[theme_id].name
                for argument_label, argument_id in event.arguments.items():
                    argument_label_no_numbers = argument_label
                    while argument_label_no_numbers.endswith(("+", "2", "3", "4", "5", "6", "7", "8")):
                        argument_label_no_numbers = argument_label_no_numbers[:-1]
                    if (not argument_label.startswith(("Theme", "Participant", event_label))):
                        if genia is True and (argument_label == "Site"):
                            if theme_label[-1] != argument_label[-1]:
                                continue
                        if argument_id.startswith("T"):
                            argument = entities[argument_id].name
                        else:
                            argument = entities[events[argument_id].arguments[events[argument_id].name]].name
                        if doc_level is True:
                            event_list[1].add((event_label + "_" + argument_label_no_numbers, (argument_label_no_numbers, argument, event_label, theme, pubmed_id)))
                        else:
                            event_list[1].add((event_label + "_" + argument_label_no_numbers, (argument_label_no_numbers, argument, event_label, theme)))
                    elif (argument_label.startswith(("Theme", "Participant"))) and (argument_label != theme_label) and (argument_id.startswith("T")):
                        if genia is True and (argument_label == "Site"):
                            if theme_label[-1] != argument_label[-1]:
                                continue
                        argument = entities[argument_id].name
                        if doc_level is True:
                            event_list[1].add((event_label + "_" + argument_label_no_numbers, (argument_label_no_numbers, argument, event_label, theme, pubmed_id)))
                        else:
                            event_list[1].add((event_label + "_" + argument_label_no_numbers, (argument_label_no_numbers, argument, event_label, theme)))

    # Question Type 3: Regulation of Simple Events
    for event in events.values():
        event_label = event.name
        if event_label in REGULATION_TYPES:
            for theme_label, theme_id in event.arguments.items():
                if theme_label.startswith(("Theme", "Participant")):
                    theme_event = event
                    protein_id = theme_id
                    if protein_id.startswith("T"):
                        continue
                    while protein_id.startswith("E"):
                        theme_event = events[protein_id]
                        protein_id = ""
                        for theme_theme_label, theme_theme_id in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                protein_id = theme_theme_id
                    if protein_id.startswith("T"):
                        for theme_theme_label, theme in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                if doc_level is True:
                                    event_list[2].add(("REGULATION", ("REGULATION", theme_event.name, proteins[theme].name, pubmed_id)))
                                else:
                                    event_list[2].add(("REGULATION", ("REGULATION", theme_event.name, proteins[theme].name)))

    # Question Type 4: Find arguments of nested regulations
    def findProteinCause(proteins, events, event, theme):
        for cause_label, cause_id in event.arguments.items():
            if cause_label.startswith(("Cause")):
                cause_event = event
                protein_id = cause_id
                while protein_id.startswith("E"):
                    cause_event = events[protein_id]
                    protein_id = ""
                    for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                        if cause_theme_label.startswith(("Theme", "Participant")):
                            protein_id = cause_theme_id
                if protein_id.startswith("T"):
                    for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                        if (cause_theme_label.startswith(("Theme", "Participant")) and cause_event.id != event.id):
                            trigger_name = entities[cause_event.arguments[cause_event.name]].name
                            if doc_level is True:
                                event_list[3].add(("Cause", ("Cause", trigger_name, "REGULATION", theme[0], theme[1], pubmed_id)))
                            else:
                                event_list[3].add(("Cause", ("Cause", trigger_name, "REGULATION", theme[0], theme[1])))
                        elif (cause_theme_label.startswith(("Cause")) and cause_event.id == event.id):
                            if doc_level is True:
                                event_list[3].add(("Cause", ("Cause", proteins[protein_id].name, "REGULATION", theme[0], theme[1], pubmed_id)))
                            else:
                                event_list[3].add(("Cause", ("Cause", proteins[protein_id].name, "REGULATION", theme[0], theme[1])))

    for event in events.values():
        event_label = event.name
        if event_label in REGULATION_TYPES:
            for theme_label, theme_id in event.arguments.items():
                if theme_label.startswith(("Theme", "Participant")):
                    theme_event = event
                    protein_id = theme_id
                    if protein_id.startswith("T"):
                        continue
                    while protein_id.startswith("E"):
                        theme_event = events[protein_id]
                        protein_id = ""
                        for theme_theme_label, theme_theme_id in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                protein_id = theme_theme_id
                    if protein_id.startswith("T"):
                        for theme_theme_label, theme in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                findProteinCause(proteins, events, event, [theme_event.name, proteins[theme].name])

    return event_list


def convert_a2File(directory, filename, genia=False, doc_level=True):
    """ Converts the .a2 file in given directory according to questions of a knowledge base evaluation.

    Question types for knowledge base evaluation on a corpus level (or document level):
    - Type 1: Discover proteins in context of events (EventType, Protein)
        - Theme/Participant Event Pair of Non-regulation types

    - Type 2: Event Argument Discovery given Type 1 tuples
        - All non-Regulation event types
        - Type 2a: One Argument of Conversions and Localizations
            - (EventType, Protein_Theme, Argument),
            e.g., (Phosphorylation, Protein_Theme, Protein_Cause)
            - Conversion and Localization event types
        - Type 2b: Cause and Site Argument Pairs for Conversions
        - Type 2c: Binding Theme/Pathway Participant Pairs

    - Type 3: Regulation Discovery
        - Type 3a: Event/Protein regulating another Event/Protein directly
                   (RegulationType, Protein_Theme, Protein_Cause), (RegulationType, EventType_Theme, EventProtein_Theme, EventType_Cause, EventProtein_Cause),
                   (RegulationType, EventType_Theme, EventProtein_Theme, Protein_Cause) and (RegulationType, Protein_Theme, EventType_Cause, EventProtein_Cause)
        - Type 3b: Protein regulating another Protein transitively
                   (Protein_Theme, Protein_Cause)
        - Type 3c: Other Protein/Event regulations transitively
                   (EventType_Theme, EventProtein_Theme, EventType_Cause, EventProtein_Cause),
                   (EventType_Theme, EventProtein_Theme, Protein_Cause) and (Protein_Theme, EventType_Cause, EventProtein_Cause)

    Parameters
    ----------
    directory : str
        The directory location as string
    filename : str
        The .a2 file name, e.g., PMID-1234.a2
    doc_level : bool
        True is document level, False is corpus level evaluation

    Returns
    -------
    list
        list of event tuples for all questions
    """

    pubmed_id = filename[:-3]
    proteins = getProteinAnnotations(directory + "/" + filename[:-3] + ".a1")
    triggers, events, equivs = getEventAnnotations(directory + "/" + filename)
    # Sevem question Types
    event_list = [set() for i in range(7)]

    # Question Type 1: Theme/Participant Event Pair of Non-regulation types
    for event in events.values():
        event_label = event.name
        if event_label not in REGULATION_TYPES:
            for argument_key, argument in event.arguments.items():
                if argument_key.startswith(("Theme", "Participant")) and argument.startswith("T"):
                    theme = proteins[argument].name
                    if doc_level is True:
                        event_list[0].add((event_label, (event_label, theme, pubmed_id)))
                    else:
                        event_list[0].add((event_label, (event_label, theme)))

    # Question Type 2a: One Argument for Conversions and Localizations
    for event in events.values():
        event_label = event.name
        if event_label not in REGULATION_TYPES:
            for theme_label, theme_id in event.arguments.items():
                if theme_label.startswith(("Theme", "Participant")) and theme_id.startswith("T"):
                    theme = proteins[theme_id].name
                    for argument_label, argument_id in event.arguments.items():
                        if (not argument_label.startswith(("Theme", "Participant", event_label))) and argument_id.startswith("T"):
                            if genia is True and (theme_label[-1].isdigit() or argument_label[-1].isdigit()):
                                if theme_label[-1] != argument_label[-1]:
                                    continue
                            if argument_id in proteins:
                                argument = proteins[argument_id].name
                            else:
                                argument = triggers[argument_id].name
                            if doc_level is True:
                                event_list[1].add((event_label + "_" + argument_label, (event_label, theme, argument, pubmed_id)))
                            else:
                                event_list[1].add((event_label + "_" + argument_label, (event_label, theme, argument)))

    # Question Type 2b: Cause and Site Arguments for Conversions
    if not genia:
        for event in events.values():
            event_label = event.name
            if event_label in CONVERSION_TYPES:
                for theme_label, theme_id in event.arguments.items():
                    if theme_label.startswith(("Theme", "Participant")) and theme_id.startswith("T"):
                        theme = proteins[theme_id].name
                        for cause_label, cause_id in event.arguments.items():
                            if cause_label.startswith("Cause") and cause_id.startswith("T"):
                                cause = proteins[cause_id].name
                                for site_label, site_id in event.arguments.items():
                                    if site_label.startswith("Site") and site_id.startswith("T"):
                                        site = proteins[site_id].name
                                        if doc_level is True:
                                            event_list[2].add((event_label, (event_label, theme, cause, site, pubmed_id)))
                                        else:
                                            event_list[2].add((event_label, (event_label, theme, cause, site)))
    else:
        event_list[2].add(("placeholder"))

    # Question Type 2c: Binding Theme/Pathway Participant Pairs
    for event in events.values():
        event_label = event.name
        if event_label in ["Binding", "Pathway"]:
            for theme1_label, theme1_id in event.arguments.items():
                if theme1_label.startswith(("Theme", "Participant")) and theme1_id.startswith("T"):
                    theme1 = proteins[theme1_id].name
                    for theme2_label, theme2_id in event.arguments.items():
                        if theme2_label.startswith(("Theme", "Participant")) and theme2_id.startswith("T") and theme1_id != theme2_id:
                            theme2 = proteins[theme2_id].name
                            themes = sorted([theme1, theme2])
                            if doc_level is True:
                                event_list[3].add((event_label, (event_label, themes[0], themes[1], pubmed_id)))
                            else:
                                event_list[3].add((event_label, (event_label, themes[0], themes[1])))

    # Question Type 3a: Event/Protein regulating another Event/Protein directly
    def findCause(proteins, events, event, theme):
        for cause_label, cause_id in event.arguments.items():
            if cause_label.startswith(("Cause")):
                if cause_id.startswith("T"):
                    cause = [proteins[cause_id].name]
                    if len(theme) > 0 and len(cause) > 0:
                        theme_string = "EventTheme" if len(theme) == 2 else "ProteinTheme"
                        cause_string = "ProteinCause"
                        if doc_level is True:
                            event_list[4].add((event_label + "_" + theme_string + "_" + cause_string, tuple([event_label] + theme + cause + [pubmed_id])))
                        else:
                            event_list[4].add((event_label + "_" + theme_string + "_" + cause_string, tuple([event_label] + theme + cause)))
                else:
                    cause_event = events[cause_id]
                    cause_event_label = cause_event.name
                    if cause_event_label not in REGULATION_TYPES:
                        for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                            if cause_theme_label.startswith(("Theme", "Participant")):
                                cause = [cause_event_label, proteins[cause_theme_id].name]
                                if len(theme) > 0 and len(cause) > 0:
                                    theme_string = "EventTheme" if len(theme) == 2 else "ProteinTheme"
                                    cause_string = "EventCause"
                                    if doc_level is True:
                                        event_list[4].add((event_label + "_" + theme_string + "_" + cause_string, tuple([event_label] + theme + cause + [pubmed_id])))
                                    else:
                                        event_list[4].add((event_label + "_" + theme_string + "_" + cause_string, tuple([event_label] + theme + cause)))

    for event in events.values():
        event_label = event.name
        if event_label in REGULATION_TYPES:
            for theme_label, theme_id in event.arguments.items():
                if theme_label.startswith(("Theme", "Participant")):
                    theme = []
                    cause = []
                    if theme_id.startswith("T"):
                        theme = [proteins[theme_id].name]
                        findCause(proteins, events, event, theme)
                    else:
                        theme_event = events[theme_id]
                        theme_event_label = theme_event.name
                        if theme_event_label not in REGULATION_TYPES:
                            for theme_theme_label, theme_theme_id in theme_event.arguments.items():
                                if theme_theme_label.startswith(("Theme", "Participant")):
                                    theme = [theme_event_label, proteins[theme_theme_id].name]
                                    findCause(proteins, events, event, theme)

    # Question Type 3b: Protein regulating another Protein transitively
    def findProteinCause(proteins, events, event, theme):
        for cause_label, cause_id in event.arguments.items():
            if cause_label.startswith(("Cause")):
                cause_event = event
                protein_id = cause_id
                while protein_id.startswith("E"):
                    cause_event = events[protein_id]
                    protein_id = ""
                    for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                        if cause_theme_label.startswith(("Theme", "Participant")):
                            protein_id = cause_theme_id
                if protein_id.startswith("T"):
                    for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                        if (cause_theme_label.startswith(("Theme", "Participant")) and cause_event.id != event.id) \
                                or (cause_theme_label.startswith(("Cause")) and cause_event.id == event.id):
                            if doc_level is True:
                                event_list[5].add(("All", (theme, cause_theme_id, pubmed_id)))
                            else:
                                event_list[5].add(("All", (theme, cause_theme_id)))

    for event in events.values():
        event_label = event.name
        if event_label in REGULATION_TYPES:
            for theme_label, theme_id in event.arguments.items():
                if theme_label.startswith(("Theme", "Participant")):
                    theme_event = event
                    protein_id = theme_id
                    while protein_id.startswith("E"):
                        theme_event = events[protein_id]
                        protein_id = ""
                        for theme_theme_label, theme_theme_id in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                protein_id = theme_theme_id
                    if protein_id.startswith("T"):
                        for theme_theme_label, theme in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                findProteinCause(proteins, events, event, theme)

    # Question Type 3c: Other Protein/Event regulations transitively
    def findEventProteinCause(proteins, events, event, theme):
        for cause_label, cause_id in event.arguments.items():
            if cause_label.startswith(("Cause")):
                cause_event = event
                protein_id = cause_id
                while protein_id.startswith("E"):
                    cause_event = events[protein_id]
                    protein_id = ""
                    for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                        if cause_theme_label.startswith(("Theme", "Participant")):
                            protein_id = cause_theme_id
                if protein_id.startswith("T"):
                    for cause_theme_label, cause_theme_id in cause_event.arguments.items():
                        if (cause_theme_label.startswith(("Theme", "Participant")) and cause_event.id != event.id) \
                                or (cause_theme_label.startswith(("Cause")) and cause_event.id == event.id):
                            if len(theme) == 1 and cause_event.name not in REGULATION_TYPES:
                                if doc_level is True:
                                    event_list[6].add(("ProteinTheme_EventCause", (theme[0], cause_event.name, cause_theme_id, pubmed_id)))
                                else:
                                    event_list[6].add(("ProteinTheme_EventCause", (theme[0], cause_event.name, cause_theme_id)))
                            elif len(theme) == 2 and cause_event.name in REGULATION_TYPES:
                                if doc_level is True:
                                    event_list[6].add(("EventTheme_ProteinCause", (theme[0], theme[1], cause_theme_id, pubmed_id)))
                                else:
                                    event_list[6].add(("EventTheme_ProteinCause", (theme[0], theme[1], cause_theme_id)))
                            elif len(theme) == 2 and cause_event.name not in REGULATION_TYPES:
                                if doc_level is True:
                                    event_list[6].add(("EventTheme_EventCause", (theme[0], theme[1], cause_event.name, cause_theme_id, pubmed_id)))
                                else:
                                    event_list[6].add(("EventTheme_EventCause", (theme[0], theme[1], cause_event.name, cause_theme_id)))

    for event in events.values():
        event_label = event.name
        if event_label in REGULATION_TYPES:
            for theme_label, theme_id in event.arguments.items():
                if theme_label.startswith(("Theme", "Participant")):
                    theme_event = event
                    protein_id = theme_id
                    while protein_id.startswith("E"):
                        theme_event = events[protein_id]
                        protein_id = ""
                        for theme_theme_label, theme_theme_id in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                protein_id = theme_theme_id
                    if protein_id.startswith("T"):
                        for theme_theme_label, theme in theme_event.arguments.items():
                            if theme_theme_label.startswith(("Theme", "Participant")):
                                if theme_event.name in REGULATION_TYPES:
                                    findEventProteinCause(proteins, events, event, [theme])
                                else:
                                    findEventProteinCause(proteins, events, event, [theme_event.name, theme])

    return event_list


def convertDirectory(directory, genia=False, doc_level=True, converter=convert_a2File):
    """ Converts all .a2 files in given directory according to custom eval format
    Parameters
    ----------
    directory : str
        The directory location as string
    doc_level : bool
        True is document level, False is corpus level evaluation

    Returns
    -------
    list
        list of event sets all for six questions
    """

    event_list = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".a2"):
            result = converter(directory, filename, genia, doc_level)
            if len(event_list) == 0:
                event_list = result
            else:
                event_list = [event_list[i] | answer_tuple for i, answer_tuple in enumerate(result)]
    return event_list


def evaluateAllQuestions(groundtruth_list, predictions_list):
    """ Prints classification report for all questions
    Parameters
    ----------
    groundtruth_list :
        list of event tuples for all questions
    predictions_list :
        list of event tuples for all questions
    """

    questions = ["Type 1: Theme/Participant Event Pairs", "Type 2a: Single Arguments for Conversions and Localizations",
                 "Type 2b: Cause and Site Argument Pairs for Conversions",
                 "Type 2c: Binding_Theme/Pathway_Participant Pairs", "Type 3a: Event/Protein regulating another Event/Protein directly",
                 "Type 3b: Protein regulating another Protein transitively", "Type 3c: Other Protein/Event regulations transitively"]
    ps = []
    rs = []
    f1s = []
    s = []
    for i, (groundtruth, prediction) in enumerate(zip(groundtruth_list, predictions_list)):
        report1, averages = getClassificationReport(groundtruth, prediction)
        print("\n====================================================================="
              + "\n\nClassification report: {}\n\n{}\n{}".format(questions[i], report1, averages[0]))
        ps.append(averages[1])
        rs.append(averages[2])
        f1s.append(averages[3])
        s.append(averages[4])
    groundtruth_set = set.union(*groundtruth_list)
    prediction_set = set.union(*predictions_list)
    total, _, _, _, _ = computeAverages(groundtruth_set, prediction_set, ps, rs, f1s, s)
    _, _, f1_total = computeMetrics(groundtruth_set, prediction_set)
    print("\n====================================================================="
          + "\n\nFinal result for all questions combined\n\n{}".format(total))
    return f1_total


if __name__ == "__main__":
    # groundtruth_directory = root_path + "/data/data/PathwayCuration/dev"
    # baseline_directory = root_path + "/working_directory_TEES/output_dev/events"

    # Pathway Curation
    # groundtruth_directory = root_path + "/data/data/PathwayCuration/dev"
    # groundtruth_directory = root_path + "/data/data/PathwayCuration/BioNLP-ST_2013_PC_development_data"
    groundtruth_directory = root_path + "/output_files/PC_dev_Gold"
    # baseline_directory = root_path + "/working_directory_TEES/output_test/events"
    # baseline_directory = root_path + "/working_directory_TEES/model_dev2/classification-test/events"
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/PC13_dev"
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/PC13_dev_ensemble"
    # baseline_directory = root_path + "/working_directory_TEES/model_dev_pc/classification-test/events"
    baseline_directory = root_path + "/output_files/PC_dev_TEES_Single"
    svm_directory = root_path + "/output_files/PC_dev_TEES_Single_SVM"
    model_directory = root_path + "/output_files/PC_dev_QA_seed_1"
    # model_directory = root_path + "/output_files/PC"

    # GENIA
    # groundtruth_directory = root_path + "/data/data/Genia11/BioNLP-ST_2011_genia_devel_data_rev1"
    # baseline_directory = root_path + "/working_directory_TEES/model_dev_genia/classification-test/events"
    # GENIA cnn
    # baseline_directory = root_path + "/output_files/GE_dev_TEES"
    # svm_directory = root_path + "/output_files/GE_dev_TEES_SVM"
    # baseline_directory = root_path + "/working_directory_TEES/predictions_cnn/GE11_dev_ensemble"
    # model_directory = root_path + "/output_files/GE_dev_QA"

    # truth = set([("Phosphorylation", ("Phosphorylation", "phosphorylates", "ACE1", 19556))])
    # pred = set([("Phosphorylation", ("Phosphorylation", "phosphorylates", "ACE1", 19556))])
    # print(getClassificationReport(truth, pred))

    # print(convert_a2File("/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/"
    #                      + "data/data/PathwayCuration/train", "PMID-1437141.txt"))

    doc_level = True
    genia = False
    # groundtruth = convertDirectory(groundtruth_directory, doc_level)
    # baseline = convertDirectory(baseline_directory, doc_level)
    # model = convertDirectory(model_directory, doc_level)
    # evaluateAllQuestions(groundtruth, baseline)
    # evaluateAllQuestions(groundtruth, model)

    # groundtruth = convertDirectory(root_path + "/output_files/PC_test_QA_debug", doc_level)
    # baseline = convertDirectory(root_path + "/output_files/PC_test_QA_debug", doc_level)
    # evaluateAllQuestions(groundtruth, baseline)

    with open(root_path + "/output_files/event_tuples_PC_dev.npy", 'rb') as f:
        results = pickle.load(f)
        groundtruth = convertDirectory(groundtruth_directory, genia, doc_level, convert_a2File2)
        groundtruth2 = convertDirectory(groundtruth_directory, genia, doc_level, convert_a2File)
        groundtruth_all = groundtruth + groundtruth2[2:4] + groundtruth2[5:6]
        # print("\n")
        # print(groundtruth[1])
        baseline = convertDirectory(baseline_directory, genia, doc_level, convert_a2File2)
        baseline2 = convertDirectory(baseline_directory, genia, doc_level, convert_a2File)
        baseline_all = baseline + baseline2[2:4] + baseline2[5:6]

        svm = convertDirectory(svm_directory, genia, doc_level, convert_a2File2)
        svm2 = convertDirectory(svm_directory, genia, doc_level, convert_a2File)
        svm_all = svm + svm2[2:4] + svm2[5:6]
        # print(baseline[1])
        model = convert_QATuples(results[1:], genia, doc_level)
        model2 = convertDirectory(model_directory, genia, doc_level)
        model_all = model + model2[2:4] + model2[5:6]
        # print(model[1])
        evaluateAllQuestions(groundtruth_all, baseline_all)
        evaluateAllQuestions(groundtruth_all, svm_all)
        evaluateAllQuestions(groundtruth_all, model_all)
