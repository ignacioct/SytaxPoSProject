from collections import defaultdict
from typing import List, Tuple, Dict, Union
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


class HMM:
    """
    Hidden Markov Model Class.

    Class that encapsulates the Hidden Markov Model architecture and implements
    a way to train, test and execute the Viterbi algorithm to perform PoS tagging.
    Implemented for Basque and Spanish.
    """

    def __init__(self, name, smooth_value=0):
        # name of the HMM
        self.name = name

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

        # Smoothing value
        self.smooth_value = smooth_value

        # Transition probability matrix
        self.A = np.full((len(self.tags), len(self.tags)), self.smooth_value)

        # Observation likelihoods
        self.B = None

        # Dictionary with the index of each word in the vocabulary
        self.vocab_dict = Dict[str, int]

        # Epsilon value to avoid division by 0
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
                B[:, i] = np.log2(B[:, i] / (sum(B[:, i])) + self.epsilon)

        self.B = B

    def fillA(self, word_appearence: List[List[Tuple[str, str]]]) -> None:
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
                self.A[i, :] = np.log2(mat[i, :] / (sum(mat[i])) + self.epsilon)

    def viterbi(
        self, sentence: Union[str, List[str]]
    ) -> Tuple[Dict[str, str], int, np.ndarray]:
        """
        Apply the Viterbi algorithm to calculate the best path and the log probability.
        By doing so, the PoS tagging of the sentence is obtained.

        Input
        -----
        sentence: str or List[str]
            Sentence to be tagged. It can be a string or a list of words.

        Returns
        -------
        tags: Array[(str,str)]
            Array with the words of the sentence and the obtained PoS tags.

        probability: float
            Calculated log probability of the best path, which correspond to the obtained tags.

        viterbi: numpy.ndarray
            Viterbi matrix in numpy array form
        """

        if type(sentence) is str:
            w = sentence.lower().split(" ")
        elif type(sentence) is list or type(sentence) is tuple:
            w = list(map(str.lower, sentence))
        else:
            print("Unexpected type: ", type(sentence))
            w = sentence

        # We will use a subset of B, only with the words that are passed in the sequence
        indeces = []
        for word in w:
            if word in self.vocab_dict.keys():
                indeces.append(self.vocab_dict[word])
            else:
                indeces.append(self.vocab_dict["[UNK]"])

        submatrix_B = self.B[:, indeces]

        # Initialize the Viterbi matrix
        viterbi_matrix = np.zeros((len(self.tags), len(w)))

        # Fill up the first column of the Viterbi matrix
        # Handling specially the - infinite cases
        viterbi_matrix[:, 0] = [
            -99999
            if submatrix_B[i, 0] < -21474836 or self.A[0, i] < -214748368
            else self.A[0, i] + submatrix_B[i, 0]
            for i in range(len(self.A))
        ]
        log_prob = 0
        # Fill in the Viterbi matrix and backpointer matrix
        for t in range(1, len(w)):
            for q in range(len(self.tags)):
                # Obtain the row with the highest probability in the previous step
                q1 = np.argmax(viterbi_matrix[:, t - 1])

                # Obtain the probability of the previous step
                max_pre = viterbi_matrix[q1, t - 1]

                # Getting the corresponding values in matrices A and B
                A_q1_q = self.A[q1, q]
                bq = submatrix_B[q, t]

                # Probability sum
                log_prob = max_pre + A_q1_q + bq

                # Filling the Viterbi matrix
                viterbi_matrix[q, t] = log_prob

        # The last probability of viterbi matrix is the los prob of the whole sentence
        pos_prob = np.max(viterbi_matrix[:, len(w) - 1])

        # Initialize tags list
        tags = []

        # Bactrack to obtain the path followed. It must be reversed aftwerwards.
        for t in range(len(w) - 1, -1, -1):
            lag = np.argmax(viterbi_matrix[:, t])
            # Add [WORD, TAG] tupla in the list
            tags.append((w[t], self.tags[lag]))

        tags.reverse()  # Reversing the position list

        return tags, pos_prob, viterbi_matrix

    def text_and_tags(
        self, conllu: List[List[Tuple[str, str]]]
    ) -> Tuple[List[str], List[str]]:
        """
        Parses List[List[[WORD, TYPE]]] to get the sentences and the PoS tags.

        Input
        -----
        conllu: List[List[Tuple[str, str]]]
            List containing all the tuples (word, type) that appears in the parsed file.

        Returns
        -------
        sentences: List[str]
            List of sentences.

        tags: List[str]
            List of PoS tags.
        """
        sentences = []
        tags = []
        # For every list get the sentence and tags
        for sublist in conllu:
            # Unzip the list of tuples to get the sentence and the tags
            sentece, tag = list(zip(*sublist))
            # Add the sentence and the tags to the lists
            sentences.append(sentece)
            tags.append(tag)

        return sentences, tags

    def train(self, path: str):
        """
        Trains the HMM model with the data in the given path.

        Input
        -----
        path: str
            Path to the .conllu file to be parsed.
        """

        # Parse the training data
        word_appearence = self.parse_conllu(path)

        # Filling tables A and B. Vocab is also obtained while filling B
        self.vocab_fillB(word_appearence)
        self.fillA(word_appearence)

    def make_pred(self, texts: List[str]) -> List[List[str]]:
        """
        Makes the prediction of the PoS tags for the given sentences.

        Input
        -----
        texts: List[str]
            List of texts to be tagged.

        Returns
        -------
        pred: List[List[str]]
            List of PoS tags for each sentence.
        """
        # For each sentence get the PoS tags calling to viterbi function.
        # Then zip the result to get only the list of tags.
        return list(map(lambda text: list(zip(*self.viterbi(text)[0]))[1], texts))

    def test(self, path: str) -> Dict[str, float]:
        """
        Tests the model with the data in the given path. Then calculate several metrics.

        Input
        -----
        path: str
            Path to the .conllu file to use it to test the model.

        Returns
        -------
        scores: Dict[str, float]
            Dictionary containing accuracy, recall, micro-averaged f1 and macro-averaged f1 score values.
        """

        # Parse the .conllu file to get the sentences and the tags
        dev_conllu = self.parse_conllu(path)

        # Get the sentences and the tags of the parsed file
        texts, gold = self.text_and_tags(dev_conllu)

        # Get the predictions of the model
        pred = self.make_pred(texts)

        # We cannot calculate the f1 score with the current gold and pred variables,
        # multilabel representation is not supported anymore in scikit.
        # We will transform it in a sparse matrix.

        gold = MultiLabelBinarizer(classes=self.tags).fit_transform(gold)
        pred = MultiLabelBinarizer(classes=self.tags).fit_transform(pred)

        accuracy_value = accuracy_score(gold, pred)
        recall_value = recall_score(gold, pred, average="micro")

        f1_micro = f1_score(gold, pred, average="micro")
        f1_macro = f1_score(gold, pred, average="macro")

        return {
            "Accuracy": accuracy_value,
            "Recall": recall_value,
            "Micro-averaged F1 score": f1_micro,
            "Macro-averaged F1 score": f1_macro,
        }

    def pos_tagging(self, text: str) -> Tuple[List[str], float]:
        """
        Tags the given sentence with the model.

        Input
        -----
        text: str
            Sentence to be tagged.

        Returns
        -------
        tags: List[str]
            List of PoS tags for the given sentence.
        """

        # Divide the sentence into words
        w = text.lower().split(" ")

        # Get the PoS tags and the log probability
        tags, log_prob, _ = self.viterbi(w)

        return tags, log_prob

    def pos_get_viterbi(self, text: str) -> np.ndarray:
        """
        Tags the given sentence with the model and outputs the Viterbi matrix

        Input
        -----
        text: str
            Sentence to be tagged.

        Returns
        -------
        viterbi: numpy.ndarray
            Viterbi matrix after the PoS tagging process
        """

        # Divide the sentence into words
        w = text.lower().split(" ")

        # Get the PoS tags and the log probability
        _, _, viterbi = self.viterbi(w)

        return viterbi
