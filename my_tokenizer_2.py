import constants
import numpy as np

def tokenize(data):
    processed_data = []
    for character in data:
        processed_data.append(
            np.float32(constants.TOKENS.index(character.lower()) / 100) )
             #constants.TOKENS.index(character.lower())) #----------RESTORE
    #processed_data = (processed_data + 71 * [0])[:71] #----------RESTORE
    processed_data = (processed_data + 71 * [np.float32(0)])[:71]
    #processed_data = np.array(processed_data)
    return processed_data

def classify(data):
    #output_class = (constants.CLASSES.index(data)) #----------RESTORE
    output_class = np.float32((constants.CLASSES.index(data) /100))
    #output_class = np.array(output_class)
    return output_class



