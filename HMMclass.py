from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np


class HMM:
    # Universal Dependencies POS tags
    self.tags = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]

    def __init__(self, input_sequence):
        # Los nombres tambien ni idea por que lo de sacar fotos a la pizarra no los no sabemos
        # Este es la matrix simetrica de tagsXtags

        # Input sequence
        self.w = input_sequence

        # Transition probability matrix
        self.A = np.zeros((len(self.tags), len(self.tags)))

        # Observation likelihoods
        self.B = np.zeros((len(self.tags), len(self.w.split())))

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

    def vocab(
        self, cases: List[List[Tuple[str, str]]], epsilon: int = 5
    ) -> Dict[str, int]:
        """
        Parses List[List[[WORD, TYPE]]] to count appearances of each word,
        if the word appears less than epsilon it is replaced by [UNK]. The function
        returns a Dict[WORD, INT] with the appearances of each word in the vocabulary.

        Input
        -----
        cases: List[List[Tuple[str, str]]]
            List containing all the tuples (word, type) that appears in the parsed file.

        epsilon: int
            Minimum appearances that a word has to have to be in the vocabulary.

        Returns
        -------
        vocab: Dict[str, int]
            Dictionary containing the vocabulary and each word appearances
        """

        dict_vocab: defaultdict[str, int] = defaultdict(
            lambda: 0
        )  # Create default dict to count appearances of each word

        # Count appearances of each word
        for sublist in cases:
            for e in sublist:
                dict_vocab[e[0].lower()] += 1

        # Replace words that appear rarely with [UNK] special token
        kont: int = 0
        vocab: Dict[str, int] = dict()

        for k, v in dict_vocab.items():
            if v <= epsilon:
                kont += v
            else:
                vocab[k] = v

        if kont != 0:
            vocab["[UNK]"] = kont

        return vocab

    def __fillA(self, word_appearence: List[List[Tuple[str, str]]]) -> None:
        """
        Fills up the transition matrix A, which contains the probability of a word being
        of certain syntactic type in relation with the class of the previous word

        Input
        -----
        word_appearance:List[List[Tuple[str,str]]]
            List containing all the tuples (word, type) that appears in the parsed file.

        """
        mat = np.zeros((len(self.tags), len(self.tags)), dtype=int)
        # For every sentence in word_appearence count the words
        for sentece in word_appearence:
            # The first prev_word is X (*)
            prev_word = "X"
            for i in range(1, len(sentece)):
                word = sentece[i][1]
                mat[self.tags.index(prev_word)][self.tags.index(word)] += 1
                prev_word = word
            # The last word is X (*)
            word = "X"
            mat[self.tags.index(prev_word)][self.tags.index(word)] += 1
        # Use the counted words to calculate the log probability
        for i, _ in enumerate(mat):
            # If the row is full of 0s then we got NaN in the division, so we put -inf before.
            if sum(mat[i]) == 0:
                self.A[i] = np.matrix([float("-inf") for w in range(len(mat[i]))])
                continue
            # Calculate the log2 probability
            for j in range(len(mat[i])):
                self.A[i][j] = np.log2(mat[i][j] / sum(mat[i]))

    def __fillB(self, word_appearance: List[List[Tuple[str, str]]]) -> None:
        """
        Fills up the matrix B, which contains the probability of a given word being
        of each syntactic class.

        Input
        -----
        word_appearance:List[List[Tuple[str,str]]]
            List containing all the tuples (word, type) that appears in the parsed file.


        """
        # Split the input sequence into words
        words = self.w.split()

        num_tags = len(self.tags)
        num_words = len(words)
        B = np.zeros((num_tags, num_words))

        # Count word occurrences given their corresponding tags
        for sentence in word_appearance:
            for word, tag in sentence:
                # If the word is in the input we count it in its index position
                if tag in self.tags and word in words:
                    tag_idx = self.tags.index(tag)
                    word_idx = words.index(word)
                    B[tag_idx][word_idx] += 1

        # # Debug print for B matrix
        # print("Emission Matrix counts:")
        # print(B)

        # Use the counted words to calculate the log probability
        for i, row in enumerate(B):
            for j, _ in enumerate(row):
                # If the column is full of 0s then we got NaN in the division, so we put -inf before.
                if sum(B[:, j]) == 0:
                    self.B[:, j] = np.matrix(
                        [float("-inf") for w in range(len(B[:, j]))]
                    )
                    continue
                # Calculate the log2 probability
                self.B[i][j] = np.log2(B[i][j] / sum(B[:, j]))

    def train(self, path):
        print("Training")
        word_appearence = self.parse_conllu(path)
        self.__fillA(word_appearence)
        self.__fillB(word_appearence)
        print(self.tags)
        print(self.A)
        print(self.B)

        # self.B = np.zeros() se tendria


hmm = HMM("estar pendiente a eso")
print(hmm.parse_conllu("UD_Spanish-AnCora/es_ancora-ud-dev.conllu"))
# print(hmm.parse_conllu("UD_Basque-BDT/eu_bdt-ud-dev.conllu"))
hmm.train("UD_Spanish-AnCora/es_ancora-ud-dev.conllu")
