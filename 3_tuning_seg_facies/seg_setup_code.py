#
# These are functions defined in notebook 1 now moved out to 
# file so that we don't use us real estate in the later notebooks
#
import h5py

def setup(filepath):
    with h5py.File(filepath, 'r') as f:
        X_train = f["train_x"][:]
        y_train = f["train_y"][:]
        group_train = f["train_groups"][:]
        train_wells = f["train_groups"].attrs["well_names"]        
        
        X_test = f["test_x"][:]
        y_test = None
#         y_test = f["test_y"]
        group_test = f["test_groups"][:]
        test_wells = f["test_groups"].attrs["well_names"]

    return X_train, y_train, group_train, X_test, y_test, group_test, (train_wells, test_wells)