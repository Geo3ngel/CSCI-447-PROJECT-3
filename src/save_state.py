# This file will handle state saving/loading logic

# IMPORTS
import pickle
import os.path
from Interface import Interface

# Basic Functions for loading json object

# Loads an object from a relevant file path
def load_state(state_path):
    state = None
    with open(state_path, 'rb') as state_file:
        state = pickle.load(state_file)
        
    return state

# Functions for saving json object. Takes in the NN to save, the desired filename (EPOCH#) and path manager to save it in the right place.
def save_state(state_object, state_filename):
    path = os.path.join(Interface.path_manager.get_save_state_dir(), state_filename)
    
    print("INTERFACE PATH:", Interface.path_manager.get_save_state_dir())
    state = None
    with open(path, 'wb') as state_file:
        pickle.dump(state_object, state_file)
        
# Set the path manager's current save folder to the current settings if it exists, or create it
def verify_save_folder(pm, db):
    save_folder_name = ""
    for layer in db.get_layers():
        save_folder_name+=str(layer)+"-"
        
    # Remove the last extra character
    save_folder_name = save_folder_name[:-1]
    
    pm.set_save_state_folder(save_folder_name)
    
    print("FOLDER NAME:", pm.get_save_state_dir())
    
    pm.make_folder_at_dir(pm.get_save_state_dir())
        
    return select_save_state(pm)
    

# Allows you to load from a save state for a given database's layer/node combination if one is present.
def select_save_state(pm):
    # Checks that save state folder contains states
    if len(pm.find_files(pm.get_save_state_dir(), "")) > 0:
        
        # Provides the option to load from an existing save state
        awns = input("\nWould you like to load from an existing save state?")
        if awns.lower() is "y":
            exists = True
            while(exists):
                print("Current save states (Epochs):", pm.find_files(pm.get_save_state_dir(), ""))
                save_state = input("Select a saved state (Epoch #):")
                # Validate the save state exists
                path = os.path.join(pm.get_save_state_dir(), save_state)
                exists = pm.validate_file(path)
                
                if exists:
                    # Load save_state object and return!
                    return load_state(path)
                else:
                    print("Invalid save state. Try again.")
        else:
            print("Beginning new Neural Net...")
    return False