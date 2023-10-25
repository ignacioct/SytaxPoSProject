from typing import List, Tuple

import numpy as np


class HMM:
    # De momento ni idea que tags poner
    tags = ["DET", "ADJ", "NOUN", "VERB", "ADV", "CONJ", "."]

    def __init__(self, input_sequence):
        # Los nombres tambien ni idea por que lo de sacar fotos a la pizarra no los no sabemos
        # Este es la matrix simetrica de tagsXtags

        # Input sequence
        self.w = input_sequence

        # Transition probability matrix
        self.A = np.zeros((len(self.tags), len(self.tags)))

        # Observation likelihoods
        self.B = np.zeros((len(self.tags), len(self.w)))

        # Initial probability distribution over states
        self.pi = 0

    def parse_conllu(self, path: str) -> List[List[Tuple[str, str]]]:
        """
        Parses the content of a conllu file and returns a list of [WORD, TYPE].

        Input
        -----
        path:str
            Path to the .conllu file to be parsed.

        Returns
        -------
        word_appearance:List[List[Tuple[str,str]]]
            List containing all the tuples (word, type) that appears in the parsed file.
        """

        # An entity starts with # sent_id = 3LB-CAST-111_C-2-s1, and ends with an empty line.
        # There are several lines inside each .conllu file, and each entity has a header.
        # There's a global header for the file.

        word_appearance = []
        exceptions = ["_", "PUNCT", "NUM", ""]

        with open(path, "r", encoding="UTF-8") as f:
            # Boolean variables for controlling parsing logic

            entity = False

            for line in f:
                # Start of an entity
                if (
                    line[:8] == "# sent_i" and entity is False
                ):  # instead of using the in keyword, more efficient.
                    entity = True
                    sublist = []

                # If the parser is checking and entity and the first line is a number,
                # we have to include a word in the list
                if entity is True and line[0].isdigit():
                    # Separating the line into tabs
                    split_line = line.split("\t")
                    if split_line[3] not in exceptions:
                        sublist.append((split_line[2], split_line[3]))

                # End of the entity
                if entity is True and line.strip() == "":
                    entity = False
                    word_appearance.append(sublist)

        return word_appearance

    def train(self, dataset):
        # TODO
        # self.B = np.zeros() se tendria
        print("Training")


hmm = HMM("jjj")
print(hmm.parse_conllu("UD_Spanish-AnCora/es_ancora-ud-dev.conllu"))
print(hmm.parse_conllu("UD_Basque-BDT/eu_bdt-ud-dev.conllu"))
