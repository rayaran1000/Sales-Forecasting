import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src.exceptions import CustomException

def save_object(file_path,obj): # Used to save an object, used by us to save pickle files
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)# Dill used to create the pickle files

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):# Function used to load the pickle files

    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
