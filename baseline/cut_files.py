""" Remove Events without themes """

# Steps to remove all Events without themes from Gold data and Predictions
#
# GOLD data:
# - Remove all Event lines beginning with E and no Participant/Theme. Store corresponding Triggers and EventIDs
# - Iterate through all stored triggers and remove them if they do not refer to any events anymore
# - Store positions of all the triggers
# - If removed EventIDs stored, begin from the top till no new events are removed
#
# TEES data:
# - Go through all stored triggers and check position
# - If matching, delete all events with entity as trigger. Store Event IDs
# - Start with recursion from GOLD data and end if nothing more can be deleted.

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
from data_processing.file_converter import getEventAnnotations

logger = logging.getLogger(__name__)


def removeEventsWithoutThemes(directory_loc, trigger_priors):
    """ Removes all Events without Themes in .a2 files. Writes output in new folder with same name but _cut suffix.
    Parameters
    ----------
    directory_loc : str
        The directory location as string
    trigger_priors : dict
        Dict of prior triggers to be deleted as well indexed by PubMedID

    Returns
    -------
    dict
        Dict of triggers that have been deleted indexed by PubMedID

    """

    triggers_deleted = {}

    for file in os.listdir(directory_loc):
        filename = os.fsdecode(file)
        if filename.endswith(".a2"):
            pubmed_id = filename[:-3]
            # logger.info(pubmed_id)
            prior_triggers = {}
            if pubmed_id in trigger_priors:
                prior_triggers = trigger_priors[pubmed_id]
            triggers, events, equivalences = getEventAnnotations(directory_loc + "/" + filename)
            triggers, events, deleted_triggers = adjustEventAnnotations(triggers, events, prior_triggers)
            # logger.info(deleted_triggers)
            triggers_deleted[pubmed_id] = deleted_triggers
            writeEventAnnotations(directory_loc + "_cut/" + filename, triggers, events, equivalences)

    return triggers_deleted


def buildTriggerIDToEventIDIndex(trigger_dict, event_dict):
    """ Build dict from trigger ID to Event ID from datastructures.datatypes.Event class """
    index = {}
    for event_id, event in event_dict.items():
        trigger_id = event.arguments[event.name]
        if trigger_id not in index:
            index[trigger_id] = set()
        index[trigger_id].add(event.id)
    return index


def buildEventIDToEventIDIndex(event_dict):
    """ Build dict from Event to Event as theme argument from datstructures.datatypes.Event class """
    index = {}
    for event_id, event in event_dict.items():
        for argument, id in event.arguments.items():
            if id.startswith("E"):
                if id not in index:
                    index[id] = set()
                index[id].add(event.id)
    return index


def adjustEventAnnotations(entity_dict, event_dict, prior_triggers):
    """ Remove events without themes from .a2 file
    Parameters
    ----------
    entity_dict : dict
        dict EventTrigger objects
    event_dict : dict
        dict of Event objects
    prior_dict : dict
        dict of to be deleted EventTrigger objects indexed by positions

    Returns
    -------
    dict
        dict of adjusted EventTrigger objects
    dict
        dict of adjusted Event objects
    dict
        dict of deleted EventTrigger objects indexed by positions

    """

    deleted_event_ids = {}
    deleted_triggers_indexed_by_position = {}
    check_trigger_ids = set()
    triggerIDToEventIDIndex = buildTriggerIDToEventIDIndex(entity_dict, event_dict)
    eventIDToEventIDIndex = buildEventIDToEventIDIndex(event_dict)
    # if len(prior_triggers) > 0:
    #     for trigger_id, trigger in list(entity_dict.items()):
    #         if (trigger.start_index, trigger.end_index) in prior_triggers \
    #                 and prior_triggers[(trigger.start_index, trigger.end_index)].class_label == trigger.class_label:
    #             del entity_dict[trigger_id]
    #             delete_event_ids_for_trigger = triggerIDToEventIDIndex[trigger_id]
    #             for event_id in delete_event_ids_for_trigger:
    #                 deleted_event_ids[event_id] = event_dict[event_id]
    #                 del event_dict[event_id]
    for id, event in list(event_dict.items()):  # Iterate over copy
        if not ("Theme" in event.arguments or "Participant" in event.arguments):
            deleted_event_ids[id] = event
            del event_dict[id]
            check_trigger_ids.add(event.arguments[event.name])
            # logger.info(deleted_event_ids)
            # logger.info(check_trigger_ids)
    all_deleted_event_ids = deleted_event_ids
    while len(deleted_event_ids) > 0:
        new_deleted_event_ids = {}
        new_check_trigger_ids = set()
        for check_trigger_id in check_trigger_ids:
            # Check if trigger only refers to deleted events. If so, delete trigger.
            check_event_ids = triggerIDToEventIDIndex[check_trigger_id]
            # logger.info(check_trigger_id)
            # logger.info(check_event_ids)
            if check_event_ids.issubset(all_deleted_event_ids.keys()):
                start_position = entity_dict[check_trigger_id].start_index
                end_position = entity_dict[check_trigger_id].end_index
                deleted_triggers_indexed_by_position[(start_position, end_position)] = entity_dict[check_trigger_id]
                del entity_dict[check_trigger_id]
        for deleted_event_id in deleted_event_ids:
            # Also delete all events which are refered by the just deleted event.
            check_event_ids = set()
            if deleted_event_id in eventIDToEventIDIndex:
                check_event_ids = eventIDToEventIDIndex[deleted_event_id]
            for id in check_event_ids:
                # Check if referenced event already deleted
                if id in event_dict:
                    new_deleted_event_ids[id] = event_dict[id]
                    new_check_trigger_ids.add(event_dict[id].arguments[event_dict[id].name])
                    del event_dict[id]
                    # logger.info(event_dict)
        deleted_event_ids = new_deleted_event_ids
        check_trigger_ids = new_check_trigger_ids
        all_deleted_event_ids.update(new_deleted_event_ids)
    # logger.info(all_deleted_event_ids)

    return entity_dict, event_dict, deleted_triggers_indexed_by_position


def writeEventAnnotations(file_a2, entity_dict, event_dict, equivalences):
    """ Write event strucutres into the specified *.a2 file
    Parameters
    ----------
    file_a2 : str
        The file location of the events (*.a2)
    entity_dict : dict
        dict EventTrigger objects
    event_dict : dict
        dict of Event objects
    equivalences : list
        list of equivalent triggers

    """

    with open(file_a2, 'w') as myfile:
        for equiv in equivalences:
            myfile.write("*\tEquiv " + equiv[0] + " " + equiv[1] + "\n")

    with open(file_a2, 'a') as myfile:
        for entity in entity_dict.values():
            if entity.class_label in EVENT_TAGS:
                myfile.write(entity.id + "\t" + entity.class_label + " " + str(entity.start_index) + " " + str(entity.end_index)
                             + "\t" + entity.name + "\n")

    with open(file_a2, 'a') as myfile:
        for event in event_dict.values():
            event_line = event.id + "\t" + event.name + ":" + event.arguments[event.name]
            for event_argument, event_argument_id in event.arguments.items():
                event_argument_truncated = event_argument
                while event_argument_truncated.endswith("+"):
                    event_argument_truncated = event_argument_truncated[:-1]
                if event_argument_truncated != event.name:
                    event_line += " " + event_argument_truncated + ":" + event_argument_id
            myfile.write(event_line + "\n")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    triggers = removeEventsWithoutThemes(root_path + "/output_files/PC_dev_Gold", {})

    _ = removeEventsWithoutThemes(root_path + "/output_files/PC_dev_TEES_Single_SVM", triggers)
