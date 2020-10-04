# datatypes.py
# Xing Wang
# Created: 08/12/2019
# Last changed: 08/03/2020

# Class files for Protein, EventTrigger and Event
# And other Pathway Curation constants


LAMBDA_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Pathway event have no theme and are not found in the model

# Pathway Curation
EVENT_TAGGING_LABELS = ['O', 'X', 'B-Regulation', 'B-Binding', 'B-Deacetylation', 'B-Degradation', 'B-Hydroxylation',
                        'B-Negative_regulation', 'B-Dissociation', 'B-Positive_regulation', 'B-Translation',
                        'B-Localization', 'B-Inactivation', 'B-Activation', 'B-Transcription', 'B-Methylation',
                        'B-Conversion', 'B-Gene_expression', 'B-Acetylation', 'B-Demethylation', 'B-Deubiquitination',
                        'B-Ubiquitination', 'B-Dephosphorylation', 'B-Phosphorylation', 'B-Transport', 'B-Pathway',
                        'I-Regulation', 'I-Binding', 'I-Deacetylation', 'I-Degradation', 'I-Hydroxylation',
                        'I-Negative_regulation', 'I-Dissociation', 'I-Positive_regulation', 'I-Translation',
                        'I-Localization', 'I-Inactivation', 'I-Activation', 'I-Transcription', 'I-Methylation',
                        'I-Conversion', 'I-Gene_expression', 'I-Acetylation', 'I-Demethylation', 'I-Deubiquitination',
                        'I-Ubiquitination', 'I-Dephosphorylation', 'I-Phosphorylation', 'I-Transport', 'I-Pathway',
                        'B-Theme', 'B-Product', 'B-Cause', 'B-Site', 'B-AtLoc', 'B-FromLoc', 'B-ToLoc', 'B-Participant',
                        'I-Theme', 'I-Product', 'I-Cause', 'I-Site', 'I-AtLoc', 'I-FromLoc', 'I-ToLoc', 'I-Participant']

# GENIA11
EVENT_TAGGING_LABELS2 = ['O', 'X', 'B-Regulation', 'B-Binding', 'B-Protein_catabolism', 'B-Negative_regulation',
                         'B-Positive_regulation', 'B-Localization', 'B-Transcription', 'B-Gene_expression',
                         'B-Phosphorylation', 'B-Theme', 'I-Regulation', 'I-Binding', 'I-Protein_catabolism',
                         'I-Negative_regulation', 'I-Positive_regulation', 'I-Localization', 'I-Transcription',
                         'I-Gene_expression', 'I-Phosphorylation', 'B-Theme', 'B-Cause', 'B-Site', 'B-CSite',
                         'B-AtLoc', 'B-ToLoc', 'I-Theme', 'I-Cause', 'I-Site', 'I-CSite', 'I-AtLoc', 'I-ToLoc']

EVENT_TAGS = ['O', 'X', 'Regulation', 'Binding', 'Deacetylation', 'Degradation', 'Hydroxylation', 'Negative_regulation',
              'Dissociation', 'Positive_regulation', 'Translation', 'Localization', 'Inactivation', 'Activation',
              'Transcription', 'Methylation', 'Conversion', 'Gene_expression', 'Acetylation', 'Demethylation',
              'Deubiquitination', 'Ubiquitination', 'Dephosphorylation', 'Phosphorylation', 'Transport', 'Pathway', 'Protein_catabolism']

ARGUMENT_TYPES = ['Product', 'Cause', 'Site', 'AtLoc', 'FromLoc', 'ToLoc', 'CSite', 'Theme']

REGULATION_TYPES = ["Positive_regulation", "Negative_regulation", "Regulation", "Activation", "Inactivation"]

NESTED_REGULATION_TYPES = ["Positive_regulation", "Negative_regulation", "Regulation"]

CONVERSION_TYPES = ["Conversion", "Phosphorylation", "Dephosphorylation", "Acetylation", "Deacetylation",
                    "Methylation", "Demethylation", "Ubiquitination", "Deubiquitination"]


class Protein:
    """Protein from an *.a1 file"""

    def __init__(self, id, class_label, start_index, end_index, name):
        self.id = id
        self.class_label = class_label
        self.start_index = start_index
        self.end_index = end_index
        self.name = name

    def __repr__(self):
        return ("ID: " + self.id + ", Name: " + self.name + ", Type: " + self.class_label
                + ", Start: " + str(self.start_index) + ", End: " + str(self.end_index))


class EventTrigger:
    """Event trigger from an *.a2 file"""

    def __init__(self, id, class_label, start_index, end_index, name):
        self.id = id
        self.class_label = class_label
        self.start_index = start_index
        self.end_index = end_index
        self.name = name

    def __repr__(self):
        return ("ID: " + self.id + ", Name: " + self.name + ", Type: " + self.class_label
                + ", Start: " + str(self.start_index) + ", End: " + str(self.end_index))


class Event:
    """Event from an *.a2 file"""

    def __init__(self, id, arguments):
        self.id = id
        self.arguments = arguments  # from type dict
        self.name = ""
        self.negation = False
        self.speculation = False

    def __repr__(self):
        return ("ID: " + self.id + ", Name: " + self.name + ", Arguments: {}".format(self.arguments.items()))


if __name__ == "__main__":
    print(len(EVENT_TAGGING_LABELS))
    print(len(LAMBDA_WEIGHTS))
