# This file will handle state saving/loading logic

# IMPORTS
import pickle

# TODO: Interface w/ the path manager somehow to know where to put the save and what to name it?

# TODO: Use the type of the object being loaded to determine how it is placed back into the project?

# Basic Functions for loading json object

def load_state(state_filename):
    state = None
    with open(state_filename, 'rb') as state_file:
        state = pickle.load(state_file)
        
    return state

# Functions for saving json object
def save_state(state_object, state_filename):
    # TODO: Consider handling state filename differently?
    state = None
    with open(state_filename, 'wb') as state_file:
        pickle.dump(state_object, state_file)
