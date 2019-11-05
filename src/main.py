""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""

# -------------------------------------------------------------
# Third-party imports

import numpy as np

# -------------------------------------------------------------
# Custom imports

import process_data
import Cost_Functions as cf
from rbf import RBF
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

# -------------------------------------------------------------

def prepare_db(database, pm):
    # Sets the selected database folder
    # in the path manager for referencing via full path.
    pm.set_current_selected_folder(database)
    # Processes the file path of the database into
    # a pre processed database ready to be used as a learning/training set.
    db = process_data.process_database_file(pm)

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

def main():
    pm = path_manager()
    selected_dbs = select_db(pm.find_folders(pm.get_databases_dir()))

    for database in selected_dbs:
            db = prepare_db(database, pm)
            k_nn = knn(100, db.get_dataset_type(), db.get_classifier_col(), db.get_classifier_attr_cols())
            # Run condensed nearest neighbor
            cnn = k_nn.condensed_nn(db.get_data())
            # Run edited nearest neighbor
            # Training data, first 90%
            td = db.get_data()[0:int(len(db.get_data()) * 0.9)]
            # Validation Data, last 10%
            vd = db.get_data()[int(len(db.get_data()) * 0.9):len(db.get_data())]
            enn = k_nn.edited_knn(td, vd)
            # Run data thru rbf net
            rbf = RBF(len(enn), 7)

            # Hardcoding the classes for now
            #TODO implement a way to get classes thru database.py

            X = process_data.shuffle_all(db.get_data(), 1)
            

            classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
            # get column vector storing correct classifications of each row
            y = np.array(db.get_data())[:,db.get_classifier_col()]
            rbf.fit(X, enn, y, db.get_dataset_type(), classes)
            print("Final Weights: ")
            print(rbf.weights)
            for i in range(10):
                print("Current point:")
                print(X[i])
                idx = rbf.predict(X[i], db.get_dataset_type(), enn)
                print("Predicted class: ", classes[idx])
                print('-------------------------------------------')

            
    
# -------------------------------------------------------------

if __name__ == '__main__':
    main()