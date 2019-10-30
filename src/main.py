""" -------------------------------------------------------------
@file        main.py
@authors     George Engel, Troy Oster, Dana Parker, Henry Soule
@brief       The file that runs the program
"""
from rbf import RBF
import process_data
from knn import knn
from kcluster import kcluster
from path_manager import pathManager as path_manager

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

# Cleaner print outs for the sake of my sanity.
def print_db(db):
    if len(db) < 1:
        print("[] - Empty")
    else:
        for row in db:
            print(row)


'''
-----------------------------------------------
Main Execution
-----------------------------------------------
'''

pm = path_manager()
selected_dbs = select_db(pm.find_folders(pm.get_databases_dir()))

for database in selected_dbs:
    db = prepare_db(database, pm)
    knn = knn(100, db.get_dataset_type(), db.get_classifier_col(), db.get_classifier_attr_cols())
    # Run condensed nearest neighbor
    cnn = knn.condensed_nn(db.get_data())

    # Run edited nearest neighbor
    td = db.get_data()[0:int(len(db.get_data()) * 0.9)] # First 90% of data
    vd = db.get_data()[int(len(db.get_data()) * 0.9):len(db.get_data())] # Last 10% of data
    enn = knn.edited_knn(td, vd)
    
    print('ENN LENGTH: ', len(enn))
    print('CNN LENGTH: ', len(cnn))
    # rbf.predict(db.get_data(), knn.condensed_nn)







