""" Extract event structures from files ans answers for posing new questions """

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
from datastructures.datatypes import EVENT_TAGS

logger = logging.getLogger(__name__)


def buildTriggerToEventIDIndex(trigger_dict, event_dict):
    """ Build dict from trigger name to Event ID from datstructures.datatypes.Event class """
    index = {}
    for event_id, event in event_dict.items():
        trigger_name = trigger_dict[event.arguments[event.name]].name.lower()
        if trigger_name not in index:
            index[trigger_name] = set()
        index[trigger_name].add(event.id)
    return index


def buildLabelToEventIDIndex(trigger_dict, event_dict):
    """ Build dict from trigger name to Event ID from datstructures.datatypes.Event class """
    index = {}
    for event_id, event in event_dict.items():
        trigger_label = trigger_dict[event.arguments[event.name]].class_label
        if trigger_label not in index:
            index[trigger_label] = set()
        index[trigger_label].add(event.id)
    return index


def buildEventToEventAsThemeIndex(event_dict):
    """ Build dict from Event to Event as theme argument from datstructures.datatypes.Event class """
    index = {}
    for event_id, event in event_dict.items():
        for argument, id in event.arguments.items():
            argument_name = argument
            while argument_name.endswith(tuple(["+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])):
                argument_name = argument_name[:-1]
            if argument_name == "Theme" and id.startswith("E"):
                if id not in index:
                    index[id] = set()
                index[id].add(event.id)
    return index


def extractProteinThemeOfEventPairs(proteins, event_triggers, events, predict=False):
    """ Extract Protein = themeOf => Event Trigger Relations,
    Ignoring equivalences for now!
    Parameters
    ----------
    proteins : dict
        dict of Protein objects
    event_triggers : dict
        dict of EventTrigger objects
    events : dict
        dict of Event objects
    predict : bool
        True if events are unknown and should be predicted. False otherwise

    Returns
    -------
    list
        List of tuples (question, subject, answers), where subject is [protein_name]
        and answers is [(event_label, EventTrigger] )
    int
        Number of answer entities
    """

    protein_trigger_pairs = {}
    number_new_answers = 0
    # Add all protein as questions, even those with no answers
    for id, protein in proteins.items():
        protein_trigger_pairs[protein.name] = ["Theme", protein.name, {}]  # "Theme" argument also denotes the "Participant" Argument for Pathway Events
    if predict is False:
        for event in events.values():
            for argument, id in event.arguments.items():
                if (argument.startswith("Theme") or argument.startswith("Participant")) and not id.startswith("E"):
                    event_trigger = event_triggers[event.arguments[event.name]]
                    protein = proteins[event.arguments[argument]].name
                    protein_trigger_pairs[protein][2][event_trigger.start_index] = (event_trigger.class_label, event_trigger)
                    number_new_answers += 1
    # tuples[0] protein_name, tuples[1] EventTrigger list
    protein_trigger_pairs = [("What are events of " + tuples[1] + "?", [tuples[1], tuples[0]],
                             [tuples[2][key] for key in sorted(tuples[2].keys())])
                             for tuples in protein_trigger_pairs.values()]
    return protein_trigger_pairs, number_new_answers


def extractEvents(proteins, triggers, events, prior_events, trigger_to_event_id_index, label_to_event_id_index, event_theme_to_event_index,
                  recursion_depth, question_type, predict=False):
    """ Extract events, Use class_label as priors instead of the triggers. For questions not including the trigger evidence
    Parameters
    ----------
    proteins : dict
        dict of Protein objects
    triggers : dict
        dict of EventTrigger objects
    events : dict
        dict of Event objects
    prior_events : list
        Previously found theme event pairs, however pubmed_id is already specified here
        [ {pubmed_id : [([subjects], [(label, label, start_index_word, end_index_word, star_index_token, end_index_token)])]} ]
    trigger_to_event_id_index : dict
        Label trigger name (lower) to Event ID dict
    label_to_event_id_index : dict
        Label trigger name (lower) to Event ID dict
    event_theme_to_event_index : dict
        Mapping of event ID which is theme to another event ID
    recursion_depth: int
        Number of recursion steps beginning from 0
    question_type : int
        0, if trigger question. 1, if event argument question
    predict : bool
        True if events are unknown and should be predicted. False otherwise

    Returns
    -------
    list
        List of tuples (question, subject, answers), where subject is [event_name, event_label, theme_protein]
        and answers is [(regulation_event_type, EventTrigger), ...]
    int
        Number of answer entities
    """

    entities = {**proteins, **triggers}
    event_arguments = {}  # Ask a question for each event tuple found in a previous step
    number_new_answers = 0

    trigger_id_to_event_id_index = {}
    for event_id, event in events.items():
        trigger_id = triggers[event.arguments[event.name]].id
        if trigger_id not in trigger_id_to_event_id_index:
            trigger_id_to_event_id_index[trigger_id] = set()
        trigger_id_to_event_id_index[trigger_id].add(event.id)

    for subject, answers in prior_events:
        if recursion_depth == 7:
            logger.info(subject)
            logger.info(answers)
        for evidence, label, _, _, _, _ in answers:
            list_of_prior_entities = []
            if label in EVENT_TAGS:
                list_of_prior_entities = [evidence, label] + list(subject)
            elif label not in EVENT_TAGS:
                # assert(len(list_of_prior_entities) >= 3) Assertion is only valid during training, during evaluation many things can happen
                list_of_prior_entities = list(subject[0:2]) + [evidence, label] + list(subject[2:])
            # Assertion when only triggers and starting_protein prevalent in the prior_event tuple
            # TODO: Change this when including event arguments into the prior_event_tuple
            assert(len(list_of_prior_entities) % 2 == 0)

            dict_key = label + "_" + label + "_" + "_".join(subject)
            current_event_ids = set()  # set of (current_event_id, root_event_id)
            matching_event_ids = set()  # set of root_event_ids
            matching = not predict  # True if predict is False. Otherwise do not try to match prior events and find subsequent answers to create training data.
            list_number = 0
            # logger.info(label)
            # logger.info(evidence)
            # logger.info(list_of_prior_entities)
            while list_number < len(list_of_prior_entities) and matching is True:
                # logger.info(list_number)
                new_event_ids = set()  # list of (new_event_id, root_event_id)
                entity = list_of_prior_entities[list_number]

                # Attention! Words are cased in the dicts but not in the prior_event tuples
                # Check if current event matches an event from the prior_event_dict
                if (list_number % 2 == 0) and (list_number == len(list_of_prior_entities) - 2):
                    matching = False
                    # Final protein theme
                    for event_ids in current_event_ids:
                        current_event_id = event_ids[0]
                        for current_argument_key, current_argument_value in events[current_event_id].arguments.items():
                            if current_argument_key.startswith(tuple(["Theme", "Participant"])) and current_argument_value.startswith("T"):
                                if entities[current_argument_value].name == entity:
                                    matching = True
                                    matching_event_ids.add(event_ids[1])
                elif list_number == 0:
                    # logger.info(entities)
                    # logger.info(events)
                    # logger.info(prior_events)
                    # logger.info(label_to_event_id_index)
                    # logger.info(entity)
                    # logger.info(list_of_prior_entities)
                    # Start of prior event, that is always an event trigger
                    matching = False
                    if entity in trigger_to_event_id_index:  # Sanity check due to edge cases like syntax errors as "ubiquitinationof" existing as triggers
                        new_event_ids = set([(event_id, event_id) for event_id in trigger_to_event_id_index[entity]])
                        if len(new_event_ids) != 0:
                            matching = True
                    elif entity in label_to_event_id_index:
                        new_event_ids = set([(event_id, event_id) for event_id in label_to_event_id_index[entity]])
                        if len(new_event_ids) != 0:
                            matching = True
                    current_event_ids = new_event_ids
                elif (list_number % 2 == 0) and (list_number < len(list_of_prior_entities) - 2):
                    matching = False
                    # Event trigger or event argument (both simple and nested possible)
                    for event_ids in current_event_ids:
                        current_event_id = event_ids[0]
                        for current_argument_key, current_argument_value in events[current_event_id].arguments.items():
                            # TODO Check correct handling of binding events and other stuff
                            argument_name = current_argument_key
                            while argument_name.endswith(tuple(["+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])):
                                argument_name = argument_name[:-1]
                            if argument_name == events[current_event_id].name:
                                continue
                            elif argument_name == list_of_prior_entities[list_number + 1]:
                                # Events with multiple themes, e.g., Bindings
                                if current_argument_value.startswith("T") and entities[current_argument_value].class_label in EVENT_TAGS \
                                        and entities[current_argument_value].class_label == entity:
                                    # New Event Label Argument
                                    matching = True
                                    new_event_ids.add(event_ids)
                                elif current_argument_value.startswith("T") and entities[current_argument_value].name == entity:
                                    # New Protein Argument
                                    matching = True
                                    new_event_ids.add(event_ids)
                                elif current_argument_value.startswith("E"):
                                    trigger_label = triggers[events[current_argument_value].arguments[events[current_argument_value].name]].class_label
                                    if trigger_label == entity:
                                        matching = True
                                        new_event_ids.add(event_ids)
                            elif current_argument_value.startswith("E"):  # nested event with another event as theme
                                trigger = triggers[events[current_argument_value].arguments[events[current_argument_value].name]]
                                if entity in [trigger.name, trigger.class_label] and trigger.class_label == list_of_prior_entities[list_number + 1]:
                                    new_event_ids.update(set([(new_event_id, event_ids[1]) for new_event_id in trigger_id_to_event_id_index[trigger.id]]))
                                    if len(new_event_ids) != 0:
                                        matching = True
                    if recursion_depth == 7:
                        logger.info(current_event_ids)
                        logger.info(list_number)
                    current_event_ids = new_event_ids

                list_number += 1

            if dict_key not in event_arguments:
                if label in EVENT_TAGS:
                    event_arguments[dict_key] = [label] + list_of_prior_entities[1:] + [{}]
                else:
                    event_arguments[dict_key] = list_of_prior_entities + [{}]
            # logger.info("Matching")
            # logger.info(matching)
            if matching:
                if recursion_depth == 7:
                    logger.info(subject)
                    logger.info(str(evidence) + " " + str(label))
                    logger.info(matching_event_ids)
                for matching_event_id in matching_event_ids:
                    if question_type == 0:
                        if matching_event_id in event_theme_to_event_index:
                            # if recursion_depth == 4:
                            #     logger.info(list_of_prior_entities)
                            #     logger.info(matching_event_id)
                            #     logger.info(event_theme_to_event_index[matching_event_id])
                            #     exit()
                            for id in event_theme_to_event_index[matching_event_id]:
                                nested_event = triggers[events[id].arguments[events[id].name]]
                                event_arguments[dict_key][-1][nested_event.start_index] = (nested_event.class_label, nested_event)
                                number_new_answers += 1
                    elif question_type == 1:
                        matching_event = events[matching_event_id]
                        for argument, id in matching_event.arguments.items():
                            # TODO Check correct handling of binding events and other stuff
                            argument_name = argument
                            while argument_name.endswith(tuple(["+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])):
                                argument_name = argument_name[:-1]

                            # Handle multiple binding/pathway arguments
                            if matching_event.name in ["Binding", "Pathway"] and argument_name in ["Theme", "Participant"]:
                                assert(id.startswith("T"))
                                # Last argument of list_of_prior_entities is Theme or Participant
                                if list_of_prior_entities[-2] != entities[id].name:  # further Theme or Participant
                                    assert(id.startswith("T"))
                                    protein_or_trigger_argument = entities[id]
                                    event_arguments[dict_key][-1][protein_or_trigger_argument.start_index] =\
                                        (argument_name, protein_or_trigger_argument)
                                    number_new_answers += 1

                            # Handle site arguments in bindings (only GENIA). Check that Theme Suffix and Site Suffix have the same number.
                            elif matching_event.name == "Binding" and argument_name == "Site":
                                theme_entity = list_of_prior_entities[-2]
                                site_belonging_to_current_theme = False
                                for theme_argument, theme_id in matching_event.arguments.items():
                                    theme_argument_name = theme_argument
                                    argument_name_temp = argument
                                    if theme_argument.startswith("Theme") and entities[theme_id].name == theme_entity:
                                        while True:
                                            if theme_argument_name[-1].isdigit() and argument_name_temp[-1].isdigit():
                                                if theme_argument_name[-1] == argument_name_temp[-1]:
                                                    theme_argument_name = theme_argument_name[:-1]
                                                    argument_name_temp = argument_name_temp[:-1]
                                                else:
                                                    break
                                            elif (not theme_argument_name[-1].isdigit()) and (not argument_name_temp[-1].isdigit()):
                                                site_belonging_to_current_theme = True
                                                break
                                            else:
                                                break
                                if site_belonging_to_current_theme is True:
                                    protein_or_trigger_argument = entities[id]
                                    event_arguments[dict_key][-1][protein_or_trigger_argument.start_index] =\
                                        (argument_name, protein_or_trigger_argument)
                                    number_new_answers += 1

                            # Default case
                            elif argument_name not in ["Theme", "Participant", matching_event.name]:
                                protein_or_trigger_argument = None
                                if id.startswith("T"):
                                    protein_or_trigger_argument = entities[matching_event.arguments[argument]]
                                else:  # id.startswith("E")
                                    temp_event_id = events[matching_event.arguments[argument]].id
                                    protein_or_trigger_argument =\
                                        triggers[events[temp_event_id].arguments[events[temp_event_id].name]]
                                event_arguments[dict_key][-1][protein_or_trigger_argument.start_index] =\
                                    (argument_name, protein_or_trigger_argument)
                                number_new_answers += 1
                # logger.info(event_arguments[dict_key])

    if recursion_depth == 7:
        logger.info(event_arguments)
        # exit()
    question_and_answers = []
    for tuples in event_arguments.values():
        question_type_string = "events" if question_type == 0 else "arguments"
        question = "What are " + question_type_string + " "
        for i, question_entity in enumerate(tuples[:-1]):
            # TODO: Revise questions when excluding triggers from the question
            assert(len(tuples[:-1]) % 2 == 0)
            if i == len(tuples[:-1]) - 2:
                question = question + "of " + question_entity + "?"
            elif i % 2 == 0:
                if tuples[:-1][i + 1] in EVENT_TAGS:
                    question = question + "of the " + question_entity + " "
                else:
                    question = question + "with the " + tuples[:-1][i + 1] + " " + question_entity + " "
        question_and_answers.append([question, tuples[:-1], [tuples[-1][key] for key in sorted(tuples[-1].keys())]])

    # logger.info(event_arguments.items())
    # logger.info(question_and_answers)
    # exit()

    return question_and_answers, number_new_answers
