""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""

# -------------------------------------------------------------
# Third-party imports

import numpy as np
import os.path
import save_state as ss

# -------------------------------------------------------------
# Custom imports

import process_data
import Cost_Functions as cf
from rbf import RBF
from FFNN import FFNN
from knn import knn
from kcluster import kcluster
from path_manager import pathManager as path_manager

# -------------------------------------------------------------

def select_db(databases):  
    if len(databases) == 0:
        print("ERROR: No databases found!")
        return False
    chosen = False
    db = ""
    chosen_dbs = []
    # Selection loop for database
    while(not chosen):
        print("\nEnter one of the databases displayed, or 'all' to run for all avalible databases.:", databases)
        db = input("Entry: ")
        print("database:", db)
        if db in databases:
            print("Selected:", db)
            chosen_dbs.append(db)
            chosen = True
        elif db.lower() == "all":
            print("Running for all Databases.")
            chosen_dbs = ["abalone", "car", "forestfires", "machine", "segmentation", "wine"]
            chosen = True
        else:
            print(db, "is an invalid entry. Try again.")
    return chosen_dbs


# Set the path manager's current save folder to the current settings if it exists, or create it
def verify_save_folder(pm, db):
    save_folder_name = ""
    for layer in db.get_layers():
        save_folder_name+=str(layer)+"-"
        
    # Remove the last extra character
    save_folder_name = save_folder_name[:-1]
    
    pm.set_save_state_folder(save_folder_name)
    
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
                    return ss.load_state(path)
                else:
                    print("Invalid save state. Try again.")
        else:
            print("Beginning new Neural Net...")
    return False

# -------------------------------------------------------------

def prepare_db(database, pm):
    # Sets the selected database folder
    # in the path manager for referencing via full path.
    pm.set_current_selected_folder(database)
    # Processes the file path of the database into
    # a pre processed database ready to be used as a learning/training set.
    db = process_data.process_database_file(pm)

    save_state = verify_save_folder(pm, db)
    
    if save_state is not False:
        # This is where we use the loaded save state object specified
        pass
    
    # output_file.write('CURRENT DATASET: ' + database + '\n')
    # debug_file.write('CURRENT DATASET: ' + database + '\n')
    # output_file.write('DATA TYPE: ' + db.get_dataset_type() + '\n')
    # debug_file.write('DATA TYPE: ' + db.get_dataset_type() + '\n')
    # Sanity checks.
    normal_data, irregular_data = process_data.identify_missing_data(db)
    corrected_data = process_data.extrapolate_data(normal_data, irregular_data, db.get_missing_symbol())
    # repaired_db is the total database once the missing values have been filled in.
    if len(corrected_data) > 0:
        repaired_db = normal_data + corrected_data
    else:
        repaired_db = normal_data
        
    db.set_data(repaired_db)
    # Convert the discrete data to type float.
    db.convert_discrete_to_float()
    # TODO: make it append the database name to the debug file aswell, so we can get every dataset when running for all of them.
    # debug_file.write('\n\nFULL DATASET: \n')
    # for row in db.get_data():
    #     debug_file.write(str(row) + '\n')
    
    return db

# -------------------------------------------------------------
# Cleaner print outs for the sake of my sanity.
def print_db(db):
    if len(db) < 1:
        print("[] - Empty")
    else:
        for row in db:
            print(row)

# -------------------------------------------------------------
# import scipy.spatial.distance.pdist
def main():
    pm = path_manager()
    selected_dbs = select_db(pm.find_folders(pm.get_databases_dir()))

    for database in selected_dbs:
        db = prepare_db(database, pm)
        k_nn = knn(10, db.get_dataset_type(), db.get_classifier_col(), db.get_classifier_attr_cols())
        classes = db.get_class_list() if db.get_dataset_type() == 'classification' else 1
        class_count = len(classes) if db.get_dataset_type() == 'classification' else 1
        print("CLASSES: ", classes)
        # X = process_data.shuffle_all(db.get_data(), 1)
        y = np.array(db.get_data())[:,db.get_classifier_col()]
            
        # ------------------------TEST CODE----------------------------
        
        X = db.get_data()
        centers = [X[1], X[10], X[21], X[32]]
        print("CENTERS: ")
        print(centers)
        # print("DISTANCES: ", pdist(centers))
        # print(sum(pdist(centers)))


        y = np.array(db.get_data())[:,db.get_classifier_col()]
        rbf = RBF(len(centers), class_count, 2)
        rbf.fit([X[0]], centers, y, db.get_dataset_type(), classes)
        

        # -------------------------------------------------------------




        # RUN K-MEANS
        # print("RUNNING K-MEANS")
        # kc = kcluster(10, 10, db.get_data(), db.get_classifier_attr_cols(), 'k-means')
        # centers = kc.get_centroids()
        # print("CENTERS: ")
        # print(centers)

        # rbf = RBF(len(centers), class_count, 10)
        # rbf.fit(X, centers, y, db.get_dataset_type(), classes)
        # print("FINAL WEIGHTS:")
        # print(rbf.weights)

        #RUN K-MEDOIDS
        # print("RUNNING K-MEDOIDS")
        # kc = kcluster(10, 10, db.get_data(), db.get_classifier_attr_cols(), 'k-medoids')
        # indices = kc.get_medoids()
        # centers = [db.get_data()[i] for i in indices]

        # print("CENTERS: ", centers)

        # rbf = RBF(len(centers), class_count)
        # rbf.fit(X, centers, y, db.get_dataset_type(), classes)
        # print("FINALS WEIGHTS:")
        # print(rbf.weights)



        # Run CNN
        # cnn = k_nn.condensed_nn(db.get_data())
        
        # Run edited nearest neighbor
        # Training data, first 90%
        # td = db.get_data()[0:int(len(db.get_data()) * 0.9)]
        # Validation Data, last 10%
        # vd = db.get_data()[int(len(db.get_data()) * 0.9):len(db.get_data())]
        # enn = k_nn.edited_knn(td, vd)
        
        # Run data thru rbf net
        # class_count = len(db.get_class_list()) if db.get_dataset_type() == 'classification' else 1 
        # rbf = RBF(len(enn), class_count, 100)
        
        # Get column vector storing correct classifications of each row
        # rbf.fit(X, enn, y, db.get_dataset_type(), db.get_class_list())
        
        # print("Final Weights: ")
        # print(rbf.weights)
        
        # for i in range(10):
        #     print("Current point:")
        #     print(X[i])
        #     print("Correct classification: ", X[i][db.get_classifier_col()])
        #     val = rbf.predict(X[i], db.get_dataset_type(), enn)
        #     if db.get_dataset_type() == 'classification':
        #         print("Predicted class: ", db.get_class_list()[val])
        #     else:
        #         print("Predicted value: ", val)
        #     print('-------------------------------------------')

            # -------------------------------------------------------------
            # FFNN stuff

            # # BEGIN classification FFNN
            # if db.get_dataset_type() == 'classification':

            #     # print(db.get_data())

            #     # BEGIN preprocessing
            #     process_data.FFNN_encoding(db)

            #     # (1) First layer (input layer) has 1 node per attribute.
            #     # (2) Hidden layers has arbitrary number of nodes.
            #     # (3) Output layer has 1 node per possible classification.
                
            #     layer_sizes = [
            #         len(db.get_attr()) - 1,     # (1)
            #         5, 5,                       # (2)
            #         len(db.get_class_list())]   # (3)

            #     # This number is arbitrary.
            #     # TODO: Tune this per dataset
            #     learning_rate = .02
                
            #     ffnn = FFNN(layer_sizes, db.get_dataset_type(), 
            #         db.get_data(), db.get_classifier_col(),
            #         learning_rate,
            #         class_list=db.get_class_list())

            # # BEGIN regression FFNN
            # elif db.get_dataset_type() == 'regression':

            #     # (1) First layer (input layer) has 1 node per attribute.
            #     # (2) Hidden layers has arbitrary number of nodes.
            #     # (3) Output layer has 1 node, just some real number.
            #     layer_sizes = [
            #         len(db.get_attr()), # (1)
            #         5, 5,               # (2)
            #         1                   # (3)
            #     ]
            
            # else:
            #     print('Database type invalid. Type = ' + db.get_dataset_type())

            
    
# -------------------------------------------------------------


if __name__ == '__main__':
    main()