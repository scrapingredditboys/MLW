# Tools for ML Workshop
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import extract_features

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

#Helper classes
def compute_offset_no_spaces(text, offset):
	count = 0
	for pos in range(offset):
		if text[pos] != " ": count +=1
	return count


def count_chars_no_special(text):
    count = 0
    special_char_list = ["#"]
    for pos in range(len(text)):
        if text[pos] not in special_char_list: count +=1
    return count


def count_length_no_special(text):
    count = 0
    special_char_list = ["#", " "]
    for pos in range(len(text)):
        if text[pos] not in special_char_list: count +=1
    return count


# Actual class to use
def get_features(df):
    # Create examples
    input_examples = df.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
        text_a = x["Text"], 
        text_b = None,
        label = "A" if x["A-coref"] else "B" if x["B-coref"] else "NEITHER"),  
        axis = 1)
    # Create result with BERT
    results = extract_features.examples_to_features(input_examples)

    # Process them to get context for A, B and pronoun
    index = df.index
    columns = ["emb_A", "emb_B", "emb_P", "label"]
    emb = pd.DataFrame(index=index, columns=columns)
    emb.index.name = "ID"

    for i in range(len(df)): # For each text
        # Get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
        P = df.loc[i,"Pronoun"].lower()
        A = df.loc[i,"A"].lower()
        B = df.loc[i,"B"].lower()

        # For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
        P_offset = compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"Pronoun-offset"])
        A_offset = compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"A-offset"])
        B_offset = compute_offset_no_spaces(df.loc[i,"Text"], df.loc[i,"B-offset"])
        # Figure out the length of A, B, not counting spaces or special characters
        A_length = count_length_no_special(A)
        B_length = count_length_no_special(B)

        # Initialize embeddings with zeros
        emb_A = np.zeros(768)
        emb_B = np.zeros(768)
        emb_P = np.zeros(768)

        # Initialize counts
        count_chars = 0
        cnt_A, cnt_B, cnt_P = 0, 0, 0

        features = pd.DataFrame(results.loc[i,"features"]) # Get the BERT embeddings for the current text
        for j in range(2,len(features)):  # Iterate over the BERT tokens for the current text; we skip over the first 2 tokens, which don't correspond to words
            token = features.loc[j,"token"]

            # See if the character count until the current token matches the offset of any of the 3 target words
            if count_chars  == P_offset: 
                # print(token)
                emb_P += np.array(features.loc[j,"layers"][0]['values'])
                cnt_P += 1
            if count_chars in range(A_offset, A_offset + A_length): 
                # print(token)
                emb_A += np.array(features.loc[j,"layers"][0]['values'])
                cnt_A +=1
            if count_chars in range(B_offset, B_offset + B_length): 
                # print(token)
                emb_B += np.array(features.loc[j,"layers"][0]['values'])
                cnt_B +=1								
            # Update the character count
            count_chars += count_length_no_special(token)
        # Taking the average between tokens in the span of A or B, so divide the current value by the count	
        emb_A /= cnt_A
        emb_B /= cnt_B

        # Work out the label of the current piece of text
        label = "NEITHER"
        if (df.loc[i,"A-coref"] == True):
            label = "A"
        if (df.loc[i,"B-coref"] == True):
            label = "B"

        # Put everything together in emb
        emb.iloc[i] = [emb_A, emb_B, emb_P, label]

    return emb
