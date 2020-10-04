""" Converts the .a* files from the BioNLP challenges to our custom .iob format """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
from transformers import BertTokenizer
from datastructures.datatypes import Protein, EventTrigger, Event
from data_processing.question_tagger import tagQuestion
from data_processing.event_extractor import (extractProteinThemeOfEventPairs, extractEvents, buildLabelToEventIDIndex,
                                             buildTriggerToEventIDIndex, buildEventToEventAsThemeIndex)

logger = logging.getLogger(__name__)
pubmed_id = ""


def readFile(file_loc):
    """ Reads the file line by line
    Parameters
    ----------
    file_loc : str
        The file location as string

    Returns
    -------
    list
        a list of strings used that are the header columns
    """

    with open(file_loc, "r") as f:
        data_lines = f.readlines()
    return data_lines


def writeFile(file_loc, bios_annotation):
    """ Writes the bios_annotation into the file
    Parameters
    ----------
    file_loc : str
        File to be written to
    bios_annotation : list
        List of Tuples (token, annotation)
    """

    with open(file_loc, "w") as f:
        for word in bios_annotation:
            f.write(word[0] + '\t' + word[1] + '\t' + word[2] + '\n')


def getWordTokenizationPosition(file_loc, tokenizer):
    """ Reads the file line by line
    Parameters
    ----------
    file_loc : str
        The file location as string
    tokenizer
        SciBert Tokenizer as default

    Returns
    -------
    list
        List of all tokens as the following list: [token, start_character, end_character]
    """

    words = []
    lines = readFile(file_loc)
    index = 0
    for line in lines:
        tokens = tokenizer.tokenize(line)
        tokens_index = 0
        current_token = tokens[tokens_index]
        if current_token[:2] == '##':
            current_token = current_token[2:]
        current_token_position = 0
        matching = False
        token_position_pair = []
        start_i = 0
        end_i = 0
        candidate_token = ""
        sub_token = False
        for i, char in enumerate(line, index):
            if char.lower() == current_token[current_token_position]:
                if not matching:
                    start_i = i
                    matching = True
                candidate_token = candidate_token + char.lower()
                current_token_position += 1
                if candidate_token == current_token:
                    if sub_token:
                        current_token = '##' + current_token
                    end_i = i
                    token_position_pair.append([current_token, start_i, end_i])
                    matching = False
                    candidate_token = ""
                    current_token_position = 0
                    tokens_index += 1
                    if tokens_index != len(tokens):
                        current_token = tokens[tokens_index]
                        sub_token = False
                        if current_token[:2] == '##':
                            current_token = current_token[2:]
                            sub_token = True
            else:  # char != current_token[current_token_position]
                matching = False
                candidate_token = ""
                current_token_position = 0
        index = i + 1
        words.extend(token_position_pair)
    return words


def getProteinAnnotations(file_a1):
    """ Parse Proteins from an *.a1 file
    ----------
    file_a1 : str
        The file location of the proteins (*.a1)

    Returns
    -------
    dict
        dict of Protein objects
    """

    proteins = {}
    lines = readFile(file_a1)
    for line in lines:
        line_data = line.split("\t")
        line_data_indices = line_data[1].split(" ")
        proteins[line_data[0]] = Protein(
            line_data[0], line_data_indices[0], int(line_data_indices[1]),
            int(line_data_indices[2]), line_data[2].strip().lower())
    return proteins


def getEventAnnotations(file_a2):
    """ Parse Events from an *.a2 file
    Find all events that a protein is a theme of
    Parameters
    ----------
    file_a2 : str
        The file location of the events (*.a2)

    Returns
    -------
    dict
        dict EventTrigger objects
    dict
        dict of Event objects
    list
        List of Tuples with equivalent Proteins
    """

    event_triggers = {}
    events = {}
    equivalences = []
    lines = readFile(file_a2)
    for line in lines:
        line_data = line.strip().split("\t")
        if line_data[0][0] == "T":  # Event Trigger
            line_data_indices = line_data[1].split(" ")
            event_triggers[line_data[0]] = EventTrigger(
                line_data[0], line_data_indices[0],
                int(line_data_indices[1]), int(line_data_indices[2]),
                line_data[2].lower())
        elif line_data[0][0] == "E":  # Event
            multiple_argument_names = ["Theme", "Site", "Product"]  # This list is dependent on the task at hand!
            multiple_argument_caches = [list(), list(), list()]
            line_data_indices = line_data[1].split(" ")
            events[line_data[0]] = Event(line_data[0], {})
            for i, argument in enumerate(line_data_indices):
                argument_name = argument.split(":")[0]
                argument_value = argument.split(":")[1]
                # Multiples of the same argument (Possible with Themes, Sites and Products in Pathway Curation/GENIA)
                # Argument name with numbers count as different arguments
                if (argument_name in events[line_data[0]].arguments):
                    index_multiple_arguments = multiple_argument_names.index(argument_name)
                    multiple_argument_caches[index_multiple_arguments].append(argument_value)
                else:
                    events[line_data[0]].arguments[argument_name] = argument_value
                if i == 0:  # Event name
                    events[line_data[0]].name = argument_name
            additional_argument_marker = "+"
            for i, additional_argument_type in enumerate(multiple_argument_names):
                for additional_argument in multiple_argument_caches[i]:
                    # Unified terminology: Instead of Theme and Theme2, we write Theme and Theme+
                    events[line_data[0]].arguments[additional_argument_type + additional_argument_marker] = additional_argument
                    additional_argument_marker += "+"
        elif line_data[0][0] == "M":  # Modification
            line_data_indices = line_data[1].split(" ")
            event_id = line_data_indices[1]
            if line_data_indices[0] == "Negation":
                events[event_id].negation = True
            elif line_data_indices[0] == "Speculation":
                events[event_id].speculation = True
            else:
                raise NameError("Nicht bekannte Eingabe " + file_a2)
        elif line_data[0][0] == "*":  # Equivalence
            line_data_indices = line_data[1].split(" ")
            equivalences.append([line_data_indices[1], line_data_indices[2]])
    return event_triggers, events, equivalences


def annotateFile(file_loc, recursion_depth=0, question_type=0, prior_events=None, predict=False,
                 tokenizer=BertTokenizer.from_pretrained(root_path + "/scibert/scibert_scivocab_uncased", do_lower_case=True)):
    """ Create question/answer pairs for all proteins in the file
    Parameters
    ----------
    file_loc : str
        The file location as string
    tokenizer
        SciBert Tokenizer as default
    recursion_depth: int
        Number of recursion steps beginning from 0
    question_type : int
        0, if trigger question. 1, if event argument question
    prior_events : list
        Previously found theme event pairs in the given file_loc, i.e. pubmed_id of following form
        [ (the_protein_subject, [(event_trigger, event_class_label, start_index, end_index), ...]), (next_protein...) ]
    predict : bool
        Whether to predict or not
    tokenizer: BertTokenizer
        Specify the tokenizer to use

    Returns
    -------
    list
        The annotated events by BERT in IOB format
        Checking for maximum sequence length not done here
    bool
        True if there are new answer entites. False otherwise.
    """

    proteins = getProteinAnnotations(file_loc[:-4] + ".a1")
    if predict is False:
        triggers, events, _ = getEventAnnotations(file_loc[:-4] + ".a2")
        trigger_index = buildTriggerToEventIDIndex(triggers, events)
        label_index = buildLabelToEventIDIndex(triggers, events)
        event_index = buildEventToEventAsThemeIndex(events)
    else:
        triggers = events = trigger_index = label_index = event_index = {}
    context = getWordTokenizationPosition(file_loc, tokenizer)
    results = None
    number_new_answers = 0
    new_answers = False
    if recursion_depth == 7:
        logger.info(file_loc[:-4])
    if recursion_depth == 0 and question_type == 0:
        results, number_new_answers = extractProteinThemeOfEventPairs(proteins, triggers, events, predict)
    else:
        results, number_new_answers = extractEvents(
            proteins, triggers, events, prior_events, trigger_index, label_index, event_index, recursion_depth, question_type, predict)
    files = []
    for quest, subjects, answers in results:
        # Only consider simple arguments, e.g., proteins here, not arguments which are events themselves
        # if recursion_depth == 4 and question_type == 0 and len(answers) > 0:
        #     logger.info(file_loc)
        #     logger.info(results)
        question = tokenizer.tokenize(quest)
        pubmed_id = file_loc[:-4].split('/')[-1]
        files.append(tagQuestion(question, context, answers, subjects, pubmed_id))
    if number_new_answers > 0:
        new_answers = True
    return files, new_answers


def generateAnnotationsFromDirectory(directory_loc, recursion_depth=0, question_type=0, prior_events=None, predict=False):
    """ Create question/answer pairs for all files in the directory
    Parameters
    ----------
    directory_loc : str
        The file location as string
    recursion_depth: int
        Number of recursion steps beginning from 0
    question_type : int
        0, if trigger question. 1, if event argument question
    predict : bool
        Whether to predict or not

    Returns
    -------
    list
        The annotated files for given question by BERT in IOB format
        Checking for maximum sequence length not done here
    bool
        True if there are new answer entites. False otherwise.
    """

    files = []
    new_answers_list = []
    for file in os.listdir(directory_loc):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            new_answers = False
            # For the event trigger question
            if prior_events is None:
                tmp_file, new_answers = annotateFile(
                    directory_loc + "/" + filename, recursion_depth, question_type, None, predict)
                files = files + tmp_file
            # For the subsequent event argument question and recursive regulation event question
            else:
                pubmed_id = file[:-4].split('/')[-1]
                if pubmed_id in prior_events:
                    tmp_file, new_answers = annotateFile(
                        directory_loc + "/" + filename, recursion_depth, question_type, prior_events[pubmed_id], predict)
                    files = files + tmp_file
            new_answers_list.append(new_answers)
    new_answers_bool = not all(value is False for value in new_answers_list)
    return files, new_answers_bool


# Main
if __name__ == "__main__":
    # print("/".join(os.path.realpath(__file__).split("\\")[:-2]))

    def recursion(id, events):
        theme_recursion = 1
        # cause_recursion = 1
        if "Theme" in events[id].arguments and events[id].arguments["Theme"][0] == "E":
            theme_recursion = 1 + recursion(events[id].arguments["Theme"], events)
        # if "Cause" in events[id].arguments and events[id].arguments["Cause"][0] == "E":
        #     cause_recursion = 1 + recursion(events[id].arguments["Cause"], events)
        # return max(theme_recursion, cause_recursion)
        return theme_recursion

    # Check how many nested levels the event structures have
    max_depth_themes = 0
    max_depth_general = 0
    for file in os.listdir("/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/PathwayCuration/"
                           + "BioNLP-ST_2013_PC_training_data"):
        filename = os.fsdecode(file)
        if filename[-1] != "2":
            continue
        print(filename)
        triggers, events, _ = getEventAnnotations("/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/"
                                                  + "PathwayCuration/BioNLP-ST_2013_PC_training_data/" + filename)
        depths = {}
        amount_events = len(events.items())
        while len(depths.items()) < amount_events:
            for event in events.values():
                if event.id in depths:
                    continue
                elif "Theme" not in event.arguments:
                    depths[event.id] = -100
                elif event.arguments["Theme"] in depths:
                    depths[event.id] = 1 + depths[event.arguments["Theme"]]
                elif event.arguments["Theme"][0] == "T":
                    depths[event.id] = 1
        # for event in events.values():
        #     depths[event.id] = recursion(event.id, events)
        if filename == "PMID-22139845.a2":
            print(depths.items())
        if max(depths.values()) > max_depth_themes:
            max_depth_themes = max(depths.values())
    print(max_depth_themes)

    _, events, _ = getEventAnnotations("/vol/fob-wbib-vol3/wbi_stud/wangxida/Studienprojekt/studienprojekt/data/data/"
                                       + "PathwayCuration/BioNLP-ST_2013_PC_development_data/PMID-18607552.a2")
    print(events)
