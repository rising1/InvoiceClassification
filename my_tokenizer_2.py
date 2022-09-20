import constants
import numpy as np

def tokenize(data):
    processed_data = []
    for character in data:
        processed_data.append(
            np.float32(constants.TOKENS.index(character.lower()) / 100) )
    processed_data = (processed_data + 71 * [np.float32(0)])[:71]
    return processed_data

def classify(data):
    output_class = np.float32((constants.CLASSES.index(data) /100))
    return output_class



