from collections import defaultdict
from typing import List, Tuple, Dict
import sys

import numpy as np


class HMM:
    def __init__(self, input_sequence):
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

        # Dictionary with the index of each tag
        self.tag_dict = {k: v for v, k in enumerate(self.tags)}

        # Input sequence
        self.w = input_sequence.split()

        # Smoothing value
        self.smooth_value = 0

        # Transition probability matrix
        self.A = np.full((len(self.tags), len(self.tags)), self.smooth_value)

        # Observation likelihoods
        self.B = None

        # Dictionary with the index of each word in the vocabulary
        self.vocab_dict = Dict[str, int]

        #Epsilon value to avoid division by 0
        self.epsilon = sys.float_info.min

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
                        sublist.append((split_line[1], split_line[3]))

                # End of the entity
                if entity is True and line.strip() == "":
                    entity = False
                    word_appearance.append(sublist)

        return word_appearance

    def vocab_fillB(self, cases: List[List[Tuple[str, str]]], epsilon: int = 5) -> None:
        """
        Parses List[List[[WORD, TYPE]]] to create a vocalulary of words. In addition,
        if the word appears less than epsilon it is replaced by [UNK].
        Vocabulary is used to create and fill B matrix (Mat[WORD][TYPE]).

        Input
        -----
        cases: List[List[Tuple[str, str]]]
            List containing all the tuples (word, type) that appears in the parsed file.

        epsilon: int
            Minimum appearances that a word has to have to be in the vocabulary.

        """

        # Create default dict to count appearances of each word
        dict_vocab: defaultdict[str, int] = defaultdict(lambda: 0)

        # Count appearances of each word
        for sublist in cases:
            for e in sublist:
                dict_vocab[e[0].lower()] += 1

        kont: int = 0
        vocab: Dict[str, int] = dict()

        # Fill the vocabulary with word:frequency
        for k, v in dict_vocab.items():
            if v <= epsilon:
                kont += v
            else:
                vocab[k] = v
        # Replace words that appear rarely with [UNK] special token
        if kont != 0:
            vocab["[UNK]"] = kont

        # Create a dictionary with the index of each word
        self.vocab_dict = {k: v for v, k in enumerate(vocab.keys())}

        # Fill B matrix
        B = np.full((len(self.tags), len(vocab)), self.smooth_value)

        for sublist in cases:
            for e in sublist:
                tag_idx = self.tag_dict[e[1]]
                if e[0].lower() in self.vocab_dict:
                    word_idx = self.vocab_dict[e[0].lower()]
                else:
                    word_idx = self.vocab_dict["[UNK]"]
                B[tag_idx][word_idx] += 1

        # Use the counted words to calculate the log probability
        for i in range(B.shape[1]):
            # If the column is full of 0s then we got NaN in the division, so we put -inf before.
            if sum(B[:, i]) == 0:
                B[:, i] = np.full((len(self.tags)), float("-inf"))
            else:
                # Calculate the log2 probability
                B[:, i] = np.log2(B[:, i] / (sum(B[:, i]))+self.epsilon)

        self.B = B

    def __fillA(self, word_appearence: List[List[Tuple[str, str]]]) -> None:
        """
        Fills up the transition matrix A, which contains the probability of a word being
        of certain syntactic type in relation with the class of the previous word

        Input
        -----
        word_appearance:List[List[Tuple[str,str]]]
            List containing all the tuples (word, type) that appears in the parsed file.

        """
        mat = np.full((len(self.tags), len(self.tags)), self.smooth_value)
        # For every sentence in word_appearence count the words
        for sentece in word_appearence:
            # The first prev_word is X (*)
            prev_word = "X"
            for i in range(1, len(sentece)):
                word = sentece[i][1]
                mat[self.tag_dict[prev_word]][self.tag_dict[word]] += 1
                prev_word = word
            # The last word is X (*)
            word = "X"
            mat[self.tag_dict[prev_word]][self.tag_dict[word]] += 1
        # Use the counted words to calculate the log probability

        for i in range(mat.shape[0]):
            # If the row is full of 0s then we got NaN in the division, so we put -inf before.
            if sum(mat[i]) == 0:
                self.A[i] = np.full((len(self.tags)), float("-inf"))
            else:
                # Calculate the log2 probability
                self.A[i, :] = np.log2(mat[i, :] / (sum(mat[i]))+self.epsilon)

    def train(self, path):
        print("Training the model")

        # Parse the training data
        word_appearence = self.parse_conllu(path)

        # Filling tables A and B. Vocab is also obtained while filling B
        self.vocab_fillB(word_appearence)
        self.__fillA(word_appearence)

    def viterbi(self) -> Tuple[Dict[str, str], int]:
        """
        Apply the Viterbi algorithm to calculate the best path and the probability.
        By doing so, the PoS tagging of the sentence is obtained.

        Returns
        -------
        tags: Array[(str,str)]
            Array with the words of the sentence and the obtained PoS tags.

        probability: float
            Calculated probability of the best path, which correspond to the obtained tags.
        """

        # We will use a subset of B, only with the words that are passed in the sequence
        indeces = []
        for word in self.w:
            if word in self.vocab_dict.keys():
                indeces.append(self.vocab_dict[word])
            else:
                indeces.append(self.vocab_dict["[UNK]"])

        submatrix_B = self.B[:, indeces]
        print(submatrix_B)

        # Initialize the Viterbi matrix
        viterbi_matrix = np.zeros((len(self.tags), len(self.w)))

        # Fill up the first column of the Viterbi matrix
        # Handling specially the - infinite cases
        viterbi_matrix[:, 0] = [
            -99999
            if submatrix_B[i, 0] < -21474836 or self.A[0, i] < -214748368
            else self.A[0, i] + submatrix_B[i, 0]
            for i in range(len(self.A))
        ]

        pos = []

        # Fill in the Viterbi matrix and backpointer matrix
        for t in range(1, len(self.w)):
            for q in range(len(self.tags)):

                # Obtain the row with the highest probability in the previous step
                q1 = np.argmax(viterbi_matrix[:, t - 1])

                # Obtain the probability of the previous step
                max_pre = viterbi_matrix[q1, t - 1]

                # Getting the corresponding values in matrices A and B
                A_q1_q = self.A[q1, q]
                bq = submatrix_B[q, t]

                # Probability sum
                lag = max_pre + A_q1_q + bq

                # Filling the Viterbi matrix
                viterbi_matrix[q, t] = lag

        # Bactrack to obtain the path followed. It must be reversed aftwerwards.
        for t in range(len(self.w) - 1, -1, -1):
            lag = np.argmax(viterbi_matrix[:, t])
            pos.append(self.tags[lag])

        tags = []  # Output tag dictionary

        pos.reverse()  # Reversing the position list

        for word, tag in zip(self.w, pos):
            tags.append((word, tag))

        return tags, float(np.max(viterbi_matrix[:, len(self.w) - 1]))

def main():
    # Proper names meeh
    hmm = HMM("Jordi")

    #hmm.train("UD_Basque-BDT/eu_bdt-ud-train.conllu")
    hmm.train("./UD_Spanish-AnCora/es_ancora-ud-train.conllu")
    print(hmm.viterbi())

if __name__ == "__main__":
    main()



