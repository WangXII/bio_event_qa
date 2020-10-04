"""Build the event structures from the question answering tuples"""

import os
import sys

if os.name == 'nt':
    root_path = "/".join(os.path.realpath(__file__).split("\\")[:-2])
else:
    root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

import logging
import bisect
from sortedcontainers import SortedDict
import networkx as nx
import pickle

from data_processing.file_converter import getProteinAnnotations
from datastructures.datatypes import Protein, EventTrigger, Event, ARGUMENT_TYPES, EVENT_TAGS, NESTED_REGULATION_TYPES, REGULATION_TYPES

logger = logging.getLogger(__name__)


def convertOutputToAStarFormat(args, data_dir, nn_output, writing=True):
    """ Converts all outputs from NN to BioNLP .a* eval format
    Writes answers into specified .a2 files. For questions not including the trigger evidence
    Parameters
    ----------
    args : dict
        Arguments for the model
    data_dir : string
        Directory with the .txt, and .a* files
    nn_output : list
        List of outputs from NN
        [ {pubmed_id : [([subjects], [(entity, label, start_index, end_index)])]} ]
    writing : bool
        Whether to write to output files. If not, then used for debugging
    """

    # All dicts are organized by PubMed IDs
    sentence_splitter_dict = {}
    entity_dict = {}
    entity_dict_cased = {}
    start_position_to_entity_name_to_entity_label_index = {}  # organized by uncased names
    entity_name_to_start_position_to_entity_id_index = {}  # organized by uncased names
    genia_argument_name_to_start_position_entity_id_index = {}  # organized by uncased names
    trigger_id_depth_index = {}

    event_dict = {}
    trigger_id_to_event_id_index = {}
    binding_graph_dict = {}

    for question_number, questions in enumerate(nn_output):
        recursion_number = question_number // 2
        logger.info(question_number)
        for pubmed_id, subjects in questions.items():
            with open(data_dir + "/" + pubmed_id + ".txt", 'r') as myfile:
                data = myfile.read()

                # Check if new PubMed ID. If yes, add its given entities from the .a1 file to our datastructures
                # and initialize the event dict
                if pubmed_id not in entity_dict:
                    entity_dict[pubmed_id] = getProteinAnnotations(data_dir + "/" + pubmed_id + ".a1")
                    entity_dict_cased[pubmed_id] = getProteinAnnotations(data_dir + "/" + pubmed_id + ".a1")
                    start_position_to_entity_name_to_entity_label_index[pubmed_id] = SortedDict()
                    entity_name_to_start_position_to_entity_id_index[pubmed_id] = {}
                    for protein in entity_dict[pubmed_id].values():

                        # Update to cased version instead of SciBERT's uncased version
                        protein_name = data[protein.start_index:protein.end_index]
                        assert(protein_name.lower() == protein.name)
                        protein.name = protein_name
                        protein_name_lower = protein_name.lower()

                        # Update indexes
                        if protein.start_index not in start_position_to_entity_name_to_entity_label_index[pubmed_id]:
                            start_position_to_entity_name_to_entity_label_index[pubmed_id][protein.start_index] = dict()
                            if protein_name_lower not in start_position_to_entity_name_to_entity_label_index[pubmed_id][protein.start_index]:
                                start_position_to_entity_name_to_entity_label_index[pubmed_id][protein.start_index][protein_name_lower] = set()
                                if protein.class_label not in start_position_to_entity_name_to_entity_label_index[pubmed_id][protein.start_index][protein_name_lower]:
                                    start_position_to_entity_name_to_entity_label_index[pubmed_id][protein.start_index][protein_name_lower].add(protein.class_label)
                        if protein_name_lower not in entity_name_to_start_position_to_entity_id_index[pubmed_id]:
                            entity_name_to_start_position_to_entity_id_index[pubmed_id][protein_name_lower] = {}
                        if protein.start_index not in entity_name_to_start_position_to_entity_id_index[pubmed_id][protein_name_lower]:
                            entity_name_to_start_position_to_entity_id_index[pubmed_id][protein_name_lower][protein.start_index] = []
                        entity_name_to_start_position_to_entity_id_index[pubmed_id][protein_name_lower][protein.start_index].append(protein.id)
                    event_dict[pubmed_id] = {}
                    trigger_id_depth_index[pubmed_id] = {}
                    trigger_id_to_event_id_index[pubmed_id] = {}
                    binding_graph_dict[pubmed_id] = {}

                    # Add sentence start and end positions denoted by '.' character
                    sentence_splitter_dict[pubmed_id] = [question_number for question_number, letter in enumerate(data, 1) if letter == "."]
                    sentence_splitter_dict[pubmed_id].insert(0, 0)

                for subject in subjects:
                    prior_event = subject[0]
                    for answer in subject[1]:
                        # Update to cased version instead of SciBERT's uncased version
                        entity_name = data[answer[4]:answer[5]]
                        entity_name_lower = entity_name.lower()
                        assert(entity_name.lower() == answer[0])
                        answer_id = ""  # One entity trigger for new event
                        answer_start = answer[4]
                        # Trigger at this position is not yet found
                        if (answer_start not in start_position_to_entity_name_to_entity_label_index[pubmed_id]) or \
                            (answer_start in start_position_to_entity_name_to_entity_label_index[pubmed_id]
                             and answer[0] not in start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]]) or \
                            (answer_start in start_position_to_entity_name_to_entity_label_index[pubmed_id]
                             and answer[0] in start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]]
                             and answer[1] not in start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]][answer[0]]
                                and question_number % 2 == 0):
                            # Proteins have to exist otherwise empty answer_id = "" in BioNLP
                            # Triggers are added to entity_dict
                            if question_number % 2 == 0 and ((question_number > 0 and answer[1] in NESTED_REGULATION_TYPES) or question_number == 0):
                                answer_id = "T" + str(len(entity_dict[pubmed_id]) + 1)
                                entity_dict[pubmed_id][answer_id] = EventTrigger(answer_id, answer[1], answer[4], answer[5], answer[0].lower())
                                entity_dict_cased[pubmed_id][answer_id] = EventTrigger(answer_id, answer[1], answer[4], answer[5], data[answer[4]:answer[5]])
                                # Update indexes
                                if (answer_start not in start_position_to_entity_name_to_entity_label_index[pubmed_id]):
                                    start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]] = dict()
                                if (entity_name_lower not in start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]]):
                                    start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]][entity_name_lower] = set()
                                if (answer[1] not in start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]]):
                                    start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]][answer[1]] = set()
                                start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]][entity_name_lower].add(answer[1])
                                start_position_to_entity_name_to_entity_label_index[pubmed_id][answer[4]][answer[1]].add(answer[1])
                                if entity_name_lower not in entity_name_to_start_position_to_entity_id_index[pubmed_id]:
                                    entity_name_to_start_position_to_entity_id_index[pubmed_id][entity_name_lower] = {}
                                if answer[4] not in entity_name_to_start_position_to_entity_id_index[pubmed_id][entity_name_lower]:
                                    entity_name_to_start_position_to_entity_id_index[pubmed_id][entity_name_lower][answer[4]] = []
                                if answer[1] not in entity_name_to_start_position_to_entity_id_index[pubmed_id]:
                                    entity_name_to_start_position_to_entity_id_index[pubmed_id][answer[1]] = {}
                                if answer[4] not in entity_name_to_start_position_to_entity_id_index[pubmed_id][answer[1]]:
                                    entity_name_to_start_position_to_entity_id_index[pubmed_id][answer[1]][answer[4]] = []
                                entity_name_to_start_position_to_entity_id_index[pubmed_id][entity_name_lower][answer[4]].append(answer_id)
                                entity_name_to_start_position_to_entity_id_index[pubmed_id][answer[1]][answer[4]].append(answer_id)
                            if args.task == "GE" and answer[1] not in ["Theme", "Cause"] and question_number % 2 == 1:
                                if pubmed_id not in genia_argument_name_to_start_position_entity_id_index:
                                    genia_argument_name_to_start_position_entity_id_index[pubmed_id] = {}  # organized by uncased names
                                if answer[1] not in genia_argument_name_to_start_position_entity_id_index[pubmed_id]:
                                    genia_argument_name_to_start_position_entity_id_index[pubmed_id][answer[1]] = {}
                                if answer[4] not in genia_argument_name_to_start_position_entity_id_index[pubmed_id][answer[1]]:
                                    # logger.info(pubmed_id)
                                    # logger.info(answer)
                                    answer_id = "T" + str(len(entity_dict[pubmed_id]) + 1)
                                    entity_dict[pubmed_id][answer_id] = Protein(answer_id, "Entity", answer[4], answer[5], answer[0].lower())
                                    entity_dict_cased[pubmed_id][answer_id] = Protein(answer_id, "Entity", answer[4], answer[5], data[answer[4]:answer[5]])
                                    genia_argument_name_to_start_position_entity_id_index[pubmed_id][answer[1]][answer[4]] = answer_id
                                else:
                                    answer_id = genia_argument_name_to_start_position_entity_id_index[pubmed_id][answer[1]][answer[4]]
                        else:
                            # Choose entity_id with right label
                            entity_ids = entity_name_to_start_position_to_entity_id_index[pubmed_id][entity_name_lower][answer[4]]
                            # Ambiguity if entity_id belongs two different classes, e.g., two different event triggers
                            # Heuristic take last added trigger, e.g., conversion and regulation trigger,
                            # then choose later added regulation trigger
                            # TODO: Improve scheme by marking complete event as answer and choose maximum spanning event
                            answer_id = entity_ids[-1]
                            # answer_id_temp = answer_id
                            for entity_id in entity_ids:
                                if entity_dict[pubmed_id][entity_id].class_label == answer[1]:
                                    answer_id = entity_id
                            # if len(entity_ids) > 1:
                            # if answer_id_temp != answer_id:
                            #     logger.info(entity_dict[pubmed_id][entity_ids[0]])
                            #     logger.info(entity_dict[pubmed_id][entity_ids[1]])
                            #     logger.info(answer_id)
                            #     logger.info(answer)
                            #     logger.info(prior_event)

                        # Add (Theme, Trigger) Event to event_dict, only (Theme) for Binding events
                        # Match the prior_event tuple with events from event_dict
                        prior_ids = []
                        prior = prior_event[0]  # Main trigger or protein of current event structure (simple and nested)
                        assert(prior in entity_name_to_start_position_to_entity_id_index[pubmed_id])

                        # Get all themes (entity or events) which match the prior event tuple as possible candidates
                        # TODO: Handle special case of binding events with several themes!
                        candidate_positions = set()
                        candidate_entity_prior_positions = entity_name_to_start_position_to_entity_id_index[pubmed_id][prior]
                        candidate_prior_event_ids = set()
                        if len(prior_event) == 2:
                            candidate_positions = candidate_entity_prior_positions.keys()
                        else:
                            for candidate_position, candidate_entity_ids in candidate_entity_prior_positions.items():
                                for candidate_entity_id in candidate_entity_ids:
                                    # Datastructure to store event trees where root is an event and root to leaf is prior event structure present in event_dict
                                    dag = {}
                                    candidate_bool = True
                                    prior_event_index = 0
                                    recursion_depth = 0
                                    if candidate_entity_id not in trigger_id_to_event_id_index[pubmed_id]:
                                        # Edge Case happens if new trigger with different label occurs the first time in a nested event
                                        candidate_bool = False
                                    else:
                                        current_event_ids = trigger_id_to_event_id_index[pubmed_id][candidate_entity_id]
                                        for current_event_id in current_event_ids:
                                            dag[(current_event_id, recursion_depth)] = [True, set(), set()]  # candidate_bool, predecessors, successors
                                    while ((candidate_bool) and (prior_event_index < len(prior_event))):
                                        next_event_ids = set()
                                        if prior_event_index % 2 == 0:
                                            candidate_bools = {}
                                            for current_event_id in current_event_ids:
                                                current_entity_label = prior_event[prior_event_index + 1]
                                                if event_dict[pubmed_id][current_event_id].name == "Pathway" and prior_event_index == len(prior_event) - 2:
                                                    current_entity_label = "Participant"
                                                current_entity_name = prior_event[prior_event_index]
                                                current_entity_id = ""
                                                # Check if trigger label matches
                                                if current_entity_label not in event_dict[pubmed_id][current_event_id].arguments:
                                                    candidate_bools[current_event_id] = False
                                                    # Special Cases Binding and Pathway Events as themes are not filled in yet
                                                    if event_dict[pubmed_id][current_event_id].name in ["Binding", "Pathway"] and question_number == 1:
                                                        # logger.info(binding_graph_dict[pubmed_id][current_event_id][1])
                                                        for participant_id in binding_graph_dict[pubmed_id][current_event_id][1].keys():
                                                            if entity_dict[pubmed_id][participant_id].name.lower() == current_entity_name:
                                                                # logger.info("Hello")
                                                                candidate_bools[current_event_id] = True
                                                else:
                                                    # Handle possible multiple arguments of the same type, e.g., theme, product or site
                                                    all_current_entity_labels = set()
                                                    entity_label_bools = []
                                                    for argument_key in event_dict[pubmed_id][current_event_id].arguments.keys():
                                                        if argument_key.startswith(current_entity_label):
                                                            # logger.info(current_entity_label)
                                                            all_current_entity_labels.add(argument_key)
                                                    for l, entity_label in enumerate(all_current_entity_labels):
                                                        current_entity_id = event_dict[pubmed_id][current_event_id].arguments[entity_label]
                                                        current_bool = True
                                                        # Nested Event Theme
                                                        if current_entity_id[0] == "E" and (prior_event_index == len(prior_event) - 2):
                                                            current_bool = False
                                                        elif current_entity_id[0] == "E" and event_dict[pubmed_id][current_event_id].name != prior_event[prior_event_index + 1]:
                                                            current_bool = False
                                                        if current_entity_id[0] == "E":
                                                            current_entity_id = event_dict[pubmed_id][current_event_id].arguments[event_dict[pubmed_id][current_event_id].name]
                                                        # Check if trigger name matches
                                                        if current_entity_name not in EVENT_TAGS and entity_dict[pubmed_id][current_entity_id].name.lower() != current_entity_name:
                                                            current_bool = False
                                                        entity_label_bools.append(current_bool)
                                                    if not all(value is False for value in entity_label_bools):
                                                        candidate_bools[current_event_id] = True
                                                    else:
                                                        candidate_bools[current_event_id] = False
                                                # Update previous parent (theme,trigger)-events if no matching
                                                if candidate_bools[current_event_id] is False:
                                                    recursive_event_ids = set([(current_event_id, recursion_depth)])
                                                    while len(recursive_event_ids) > 0:
                                                        recursive_event_id = recursive_event_ids.pop()
                                                        # if recursive_event_id not in dag:
                                                        # logger.info(dag)
                                                        # logger.info(pubmed_id)
                                                        # logger.info(recursive_event_id)
                                                        dag[recursive_event_id][0] = False
                                                        for predecessor_event_id in dag[recursive_event_id][1]:
                                                            all_pred_successors_bool = []
                                                            for pred_successor in dag[predecessor_event_id][2]:
                                                                all_pred_successors_bool.append(dag[pred_successor][0])
                                                            predecessor_bool = not all(value is False for value in all_pred_successors_bool)
                                                            if predecessor_bool is False:
                                                                recursive_event_ids.add(predecessor_event_id)
                                                # Update candidate_event_id in nested events
                                                if (prior_event_index < len(prior_event) - 4) and (candidate_bools[current_event_id] is True) \
                                                        and (prior_event[prior_event_index + 3] not in ARGUMENT_TYPES):
                                                    next_event_name = prior_event[prior_event_index + 2]
                                                    next_entity_id_sets = entity_name_to_start_position_to_entity_id_index[pubmed_id][next_event_name].values()
                                                    next_entity_ids = set()
                                                    for next_entity_id_set in next_entity_id_sets:
                                                        for next_entity_id in next_entity_id_set:
                                                            if next_entity_id != answer_id or answer[1] not in EVENT_TAGS:
                                                                next_entity_ids.add(next_entity_id)
                                                    for next_entity_id in next_entity_ids:
                                                        if next_entity_id not in trigger_id_to_event_id_index[pubmed_id]:
                                                            logger.info(entity_dict[pubmed_id])
                                                            logger.info(event_dict[pubmed_id])
                                                            logger.info(trigger_id_to_event_id_index[pubmed_id])
                                                            logger.info(pubmed_id)
                                                            logger.info(next_entity_id)
                                                            logger.info(answer)
                                                            logger.info(prior_event)
                                                        for next_event_id in trigger_id_to_event_id_index[pubmed_id][next_entity_id]:
                                                            # next_event_id must be argument to current_event_id
                                                            if next_event_id in event_dict[pubmed_id][current_event_id].arguments.values():
                                                                if (next_event_id, recursion_depth + 1) not in dag:
                                                                    dag[(next_event_id, recursion_depth + 1)] = [True, set(), set()]
                                                                next_event_ids.add(next_event_id)
                                                                dag[(next_event_id, recursion_depth + 1)][1].add((current_event_id, recursion_depth))
                                                                dag[(current_event_id, recursion_depth)][2].add((next_event_id, recursion_depth + 1))
                                                elif (candidate_bools[current_event_id] is True):
                                                    next_event_ids.add(current_event_id)
                                            # If all candidate_bools false then candidate_bool false
                                            candidate_bool = not all(value is False for value in candidate_bools.values())

                                            if (prior_event_index < len(prior_event) - 4) and (prior_event[prior_event_index + 3] not in ARGUMENT_TYPES):
                                                recursion_depth += 1

                                        else:
                                            next_event_ids = current_event_ids
                                        current_event_ids = next_event_ids
                                        prior_event_index += 1

                                    if candidate_bool:
                                        candidate_positions.add(candidate_position)
                                        for dag_event, dag_properties in dag.items():
                                            if dag_properties[0] is True:
                                                candidate_prior_event_ids.add(dag_event[0])

                        # Debug
                        if len(candidate_positions) == 0 or (len(prior_event) > 2 and len(candidate_prior_event_ids) == 0):
                            logger.info(entity_dict[pubmed_id])
                            logger.info(event_dict[pubmed_id])
                            logger.info(trigger_id_to_event_id_index[pubmed_id])
                            logger.info(entity_name_to_start_position_to_entity_id_index[pubmed_id])
                            # logger.info(binding_graph_dict[pubmed_id])
                            logger.info(prior_event)
                            logger.info(answer)
                            logger.info(pubmed_id)
                            continue
                        assert(len(candidate_positions) > 0)
                        assert(len(prior_event) == 2 or len(candidate_prior_event_ids) > 0)

                        # Extract nearest trigger/event, e.g., from the same sentence as the answer, and extract ID
                        sentence_index = bisect.bisect_right(sentence_splitter_dict[pubmed_id], answer[5])
                        assert(sentence_index > 0)
                        sentence_start = sentence_splitter_dict[pubmed_id][sentence_index - 1]
                        sentence_end = sentence_splitter_dict[pubmed_id][sentence_index]
                        candidate_positions_same_sentence = []
                        candidate_positions_else = []
                        for candidate_position in candidate_positions:
                            if candidate_position > sentence_start and candidate_position < sentence_end:
                                candidate_positions_same_sentence.append(candidate_position)
                            else:
                                candidate_positions_else.append(candidate_position)
                        candidate_array = []
                        if len(candidate_positions_same_sentence) >= 1:
                            candidate_array = candidate_positions_same_sentence
                        else:
                            candidate_array = candidate_positions_else
                        best_candidate_pos = -1
                        distance_theme_trigger = sys.maxsize
                        for pos in candidate_array:
                            if abs(pos - answer_start) < distance_theme_trigger:
                                best_candidate_pos = pos
                                distance_theme_trigger = abs(pos - answer_start)
                        prior_entity_ids = []
                        for prior_entity_id_with_label in entity_name_to_start_position_to_entity_id_index[pubmed_id][prior][best_candidate_pos]:
                            if len(prior_event) == 2 or entity_dict[pubmed_id][prior_entity_id_with_label].class_label == prior_event[1]:
                                prior_entity_ids.append(prior_entity_id_with_label)
                        assert(len(prior_entity_ids) > 0)
                        if len(prior_event) == 2:
                            prior_ids.append(prior_entity_ids[0])
                        else:  # Nested Regulation Trigger Question
                            for prior_entity_id in prior_entity_ids:
                                for prior_event_id in trigger_id_to_event_id_index[pubmed_id][prior_entity_id]:
                                    if prior_event_id in candidate_prior_event_ids:
                                        prior_ids.append(prior_event_id)

                        # Debug
                        if len(prior_ids) == 0:
                            logger.info(entity_dict[pubmed_id])
                            logger.info(event_dict[pubmed_id])
                            logger.info(trigger_id_to_event_id_index[pubmed_id])
                            logger.info(binding_graph_dict[pubmed_id])
                            logger.info(entity_name_to_start_position_to_entity_id_index[pubmed_id])
                            logger.info(prior_event)
                            logger.info(answer)
                            logger.info(pubmed_id)
                            logger.info(prior_entity_id)
                            logger.info(candidate_prior_event_ids)
                            logger.info(candidate_positions)
                            logger.info(best_candidate_pos)
                        assert(len(prior_ids) != 0)

                        # Add new events to event_dict
                        for prior_id_number, prior_id in enumerate(prior_ids):
                            if question_number % 2 == 0:
                                # Special treatment for Binding/Pathway events
                                # Build a graph dict to see which entity references which other entities
                                # Create events after question_number 1
                                if answer[1] in ["Binding", "Pathway"] and question_number == 0:
                                    # Found new Binding or Pathway Trigger for the first time
                                    event_id = ""
                                    if answer_id not in trigger_id_to_event_id_index[pubmed_id]:
                                        new_event_id = "E" + str(len(event_dict[pubmed_id]) + 1)
                                        event_dict[pubmed_id][new_event_id] = Event(new_event_id, {})
                                        event_dict[pubmed_id][new_event_id].name = answer[1]
                                        event_dict[pubmed_id][new_event_id].arguments[answer[1]] = answer_id
                                        if answer_id not in trigger_id_to_event_id_index[pubmed_id]:
                                            trigger_id_to_event_id_index[pubmed_id][answer_id] = set()
                                        trigger_id_to_event_id_index[pubmed_id][answer_id].add(new_event_id)
                                        if answer_id not in trigger_id_depth_index[pubmed_id]:
                                            trigger_id_depth_index[pubmed_id][answer_id] = set()
                                        trigger_id_depth_index[pubmed_id][answer_id].add(recursion_number)
                                        event_id = new_event_id
                                    else:
                                        event_id = list(trigger_id_to_event_id_index[pubmed_id][answer_id])[0]
                                    # Collect entities for Binding/Pathway trigger answer_id
                                    if event_id not in binding_graph_dict[pubmed_id]:
                                        binding_graph_dict[pubmed_id][event_id] = [set(), {}]
                                    if prior_id not in binding_graph_dict[pubmed_id][event_id][1]:
                                        binding_graph_dict[pubmed_id][event_id][1][prior_id] = ["", set([prior_id])]
                                # Add theme to new event structure
                                else:
                                    if question_number >= 2 and answer[1] not in NESTED_REGULATION_TYPES:
                                        continue
                                    # Check if event already exists if nested Regulation and Binding/Pathway as theme
                                    existing = False
                                    if answer_id in trigger_id_to_event_id_index[pubmed_id]:
                                        for existing_event_id in trigger_id_to_event_id_index[pubmed_id][answer_id]:
                                            if event_dict[pubmed_id][existing_event_id].arguments["Theme"] == prior_id:
                                                existing = True
                                    if existing is False:
                                        new_event_id = "E" + str(len(event_dict[pubmed_id]) + 1)
                                        event_dict[pubmed_id][new_event_id] = Event(new_event_id, {})
                                        event_dict[pubmed_id][new_event_id].name = answer[1]
                                        event_dict[pubmed_id][new_event_id].arguments[answer[1]] = answer_id
                                        if answer_id not in trigger_id_to_event_id_index[pubmed_id]:
                                            trigger_id_to_event_id_index[pubmed_id][answer_id] = set()
                                        trigger_id_to_event_id_index[pubmed_id][answer_id].add(new_event_id)
                                        event_dict[pubmed_id][new_event_id].arguments["Theme"] = prior_id
                                        if answer_id not in trigger_id_depth_index[pubmed_id]:
                                            trigger_id_depth_index[pubmed_id][answer_id] = set()
                                        trigger_id_depth_index[pubmed_id][answer_id].add(recursion_number)
                            else:
                                assert(prior_id[0] == "E")  # Event and not and entity "T"
                                # Only happens in question number 1
                                if answer_id != "" and event_dict[pubmed_id][prior_id].name in ["Binding", "Pathway"] and question_number == 1:
                                    # event_trigger_id = event_dict[pubmed_id][prior_id].arguments[event_dict[pubmed_id][prior_id].name]
                                    if answer[1] != "Product":
                                        participant_name = prior_event[-2]
                                        for participant_id in list(binding_graph_dict[pubmed_id][prior_id][1]):  # Force copy of dict
                                            if entity_dict[pubmed_id][participant_id].name.lower() == participant_name:
                                                if answer[1] in ["Theme", "Participant"] and entity_dict[pubmed_id][answer_id].class_label not in EVENT_TAGS:
                                                    binding_graph_dict[pubmed_id][prior_id][1][participant_id][1].add(answer_id)
                                                    if answer_id not in binding_graph_dict[pubmed_id][prior_id][1]:
                                                        binding_graph_dict[pubmed_id][prior_id][1][answer_id] = ["", set([answer_id])]
                                                    binding_graph_dict[pubmed_id][prior_id][1][answer_id][1].add(participant_id)
                                                elif answer[1] == "Site":
                                                    binding_graph_dict[pubmed_id][prior_id][1][participant_id][0] = answer_id
                                    else:
                                        binding_graph_dict[pubmed_id][prior_id][0].add(answer[1])
                                elif answer_id != "" and entity_dict[pubmed_id][answer_id].class_label not in EVENT_TAGS:
                                    # Check if ToLoc/FromLoc/AtLoc is Entity or Cellular_component and CSite/Site is Entity or Simple_chemical
                                    if answer[1] in ["ToLoc", "AtLoc", "FromLoc"] and entity_dict[pubmed_id][answer_id].class_label not in ["Cellular_component", "Entity"]:
                                        continue
                                    elif answer[1] in ["CSite", "Site"] and entity_dict[pubmed_id][answer_id].class_label not in ["Simple_chemical", "Entity"]:
                                        continue
                                    elif answer[1] != "Cause" and event_dict[pubmed_id][prior_id].name in REGULATION_TYPES:
                                        continue
                                    elif question_number > 1 and event_dict[pubmed_id][prior_id].name not in NESTED_REGULATION_TYPES:
                                        continue
                                    # Check if argument already exists. If so, duplicate event and replace argument
                                    if answer[1] not in event_dict[pubmed_id][prior_id].arguments:
                                        event_dict[pubmed_id][prior_id].arguments[answer[1]] = answer_id
                                    else:
                                        if prior_id_number + 1 == len(prior_ids):
                                            # Check if the same event as prior_id with answer_id as argument for answer[1] already exists
                                            existing = False
                                            for check_event in event_dict[pubmed_id].values():
                                                matching_arguments = True
                                                for check_argument, check_argument_value in check_event.arguments.items():
                                                    if answer[1] == check_argument and check_argument in event_dict[pubmed_id][prior_id].arguments:
                                                        if check_argument_value != answer_id:
                                                            matching_arguments = False
                                                    elif check_argument in event_dict[pubmed_id][prior_id].arguments:
                                                        if check_argument_value != event_dict[pubmed_id][prior_id].arguments[check_argument]:
                                                            matching_arguments = False
                                                    elif check_argument not in event_dict[pubmed_id][prior_id].arguments:
                                                        matching_arguments = False
                                                if matching_arguments is True:
                                                    existing = True
                                                    break

                                            if existing is False:
                                                new_event_id = "E" + str(len(event_dict[pubmed_id]) + 1)
                                                new_event_name = event_dict[pubmed_id][prior_id].name
                                                event_dict[pubmed_id][new_event_id] = Event(new_event_id, {})
                                                event_dict[pubmed_id][new_event_id].name = new_event_name
                                                for argument_key, argument_value in event_dict[pubmed_id][prior_id].arguments.items():
                                                    event_dict[pubmed_id][new_event_id].arguments[argument_key] = argument_value
                                                event_dict[pubmed_id][new_event_id].arguments[answer[1]] = answer_id
                                                # Add new event to known trigger_id
                                                event_trigger_id = event_dict[pubmed_id][prior_id].arguments[new_event_name]
                                                if event_trigger_id not in trigger_id_to_event_id_index[pubmed_id]:
                                                    logger.info(trigger_id_to_event_id_index[pubmed_id])
                                                    logger.info(entity_dict[pubmed_id])
                                                    logger.info(event_dict[pubmed_id])
                                                    logger.info(pubmed_id)
                                                    logger.info(event_trigger_id)
                                                    logger.info(prior_id)
                                                trigger_id_to_event_id_index[pubmed_id][event_trigger_id].add(new_event_id)
                                elif answer_id != "" and entity_dict[pubmed_id][answer_id].class_label in EVENT_TAGS:  # Whole event as cause of a regulation event
                                    # TODO: Adapt for whole event marked as cause
                                    cause_event_ids = trigger_id_to_event_id_index[pubmed_id][answer_id]
                                    for cause_event_id in cause_event_ids:
                                        if cause_event_id != prior_id:
                                            if answer[1] != "Cause":
                                                logger.info(entity_dict[pubmed_id][answer_id])
                                                logger.info(answer)
                                                logger.info(prior_event)
                                            # assert(answer[1] == "Cause")

                                            if answer[1] == "Cause":
                                                # Prevent recursion loops
                                                def checkrecursion(event_dict, cause_event_id, prior_id):
                                                    for argument_cause_event_key, argument_cause_event in event_dict[cause_event_id].arguments.items():
                                                        if argument_cause_event.startswith("E"):
                                                            if argument_cause_event == prior_id:
                                                                return True
                                                            else:
                                                                checkrecursion(event_dict, argument_cause_event, prior_id)
                                                    return False

                                                recursion_loop = checkrecursion(event_dict[pubmed_id], cause_event_id, prior_id)
                                                if recursion_loop is False:
                                                    event_dict[pubmed_id][prior_id].arguments[answer[1]] = cause_event_id
                                # else:
                                #     logger.info(entity_dict[pubmed_id])
                                #     logger.info(event_dict[pubmed_id])
                                #     logger.info("PubMedID: " + pubmed_id + ", Answer ID: " + str(answer_id) + ", Prior ID:" + str(prior_id))
                                #     logger.info("Event Tuple" + str(subject))
                                #     exit()

        # Handle Binding and Pathway Events
        # Find strongly connected components
        if question_number == 1:
            for pubmed_id, binding_dict in binding_graph_dict.items():
                for event_id, binding_arguments in binding_dict.items():
                    trigger_id = event_dict[pubmed_id][event_id].arguments[event_dict[pubmed_id][event_id].name]
                    products, protein_dict = binding_arguments
                    event_name = entity_dict[pubmed_id][trigger_id].class_label
                    argument_name = "Theme" if event_name == "Binding" else "Participant"
                    graph = nx.DiGraph()
                    graph.add_nodes_from(protein_dict.keys())
                    for protein_id, arguments in protein_dict.items():
                        for neighbour in arguments[1]:
                            graph.add_edge(protein_id, neighbour)
                    undirected_graph = graph.to_undirected(True)
                    components = nx.find_cliques(undirected_graph)
                    for i, component in enumerate(components):
                        comp_event_id = event_id
                        if i >= 1:
                            comp_event_id = "E" + str(len(event_dict[pubmed_id]) + 1)
                            event_dict[pubmed_id][comp_event_id] = Event(comp_event_id, {})
                            event_dict[pubmed_id][comp_event_id].name = event_name
                            event_dict[pubmed_id][comp_event_id].arguments[event_name] = trigger_id
                            if trigger_id not in trigger_id_to_event_id_index[pubmed_id]:
                                trigger_id_to_event_id_index[pubmed_id][trigger_id] = set()
                            trigger_id_to_event_id_index[pubmed_id][trigger_id].add(comp_event_id)
                        product_index = 0
                        for product in products:
                            product_argument = "Product"
                            if product_index >= 1:
                                product_argument = "Product" + str(product_index + 1)
                            event_dict[pubmed_id][comp_event_id].arguments[product_argument] = product
                        for j, comp_protein_id in enumerate(component):
                            binding_argument = argument_name
                            if j >= 1:
                                binding_argument = argument_name + str(j + 1)
                            event_dict[pubmed_id][comp_event_id].arguments[binding_argument] = comp_protein_id
                            if protein_dict[comp_protein_id][0] != "":  # Site Argument exists (only in GENIA)
                                site_id = protein_dict[comp_protein_id][0]
                                site_argument = "Site"
                                if j >= 1:
                                    site_argument = "Site" + str(j + 1)
                                event_dict[pubmed_id][comp_event_id].arguments[site_argument] = site_id

    # logger.info(entity_dict["PMID-8868471"].items())
    # logger.info(event_dict["PMID-8868471"].items())
    # exit()

    if writing is True:
        for pubmed_id in entity_dict_cased:
            with open(args.predictions_dir + args.task + "/" + pubmed_id + ".a2", 'w') as myfile:
                for entity in entity_dict_cased[pubmed_id].values():
                    if entity.class_label in EVENT_TAGS or entity.class_label == "Entity":
                        myfile.write(entity.id + "\t" + entity.class_label + " " + str(entity.start_index) + " " + str(entity.end_index)
                                     + "\t" + entity.name + "\n")
        for pubmed_id in event_dict:
            with open(args.predictions_dir + args.task + "/" + pubmed_id + ".a2", 'a') as myfile:
                for event in event_dict[pubmed_id].values():
                    event_line = event.id + "\t" + event.name + ":" + event.arguments[event.name]
                    for event_argument, event_argument_id in event.arguments.items():
                        if event_argument != event.name:
                            event_line += " " + event_argument + ":" + event_argument_id
                    myfile.write(event_line + "\n")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logger.addHandler(console)
    # with open(root_path + "/output_files/event_tuples_without_triggers.npy", 'rb') as f:
    #     results = pickle.load(f)
    #     convertOutputToAStarFormat(results[0], results[0].dev_data, results[1:])
    with open(root_path + "/output_files/event_tuples_GE_test.npy", 'rb') as f:
        results = pickle.load(f)
        convertOutputToAStarFormat(results[0], results[0].test_data, results[1:], writing=True)
