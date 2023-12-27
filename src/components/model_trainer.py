import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Splitting Training and Test Input data") # Splitting the data into train and test 
            
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
           
            models = { 
                    
                'Random Forest Regressor' : RandomForestRegressor(),
                'Ridge Regression' : Ridge(),
                'Lasso Regression' : Lasso(),
                'ElasticNet Regression' : ElasticNet(),
                'Linear Regression' : LinearRegression(),
                'Support Vector Regressor' : SVR(),
                'K Neighbours Regressor' : KNeighborsRegressor(),
                'Decision Tree Regressor' : DecisionTreeRegressor(),
                'Gradient Boosting Regressor' : GradientBoostingRegressor(),
                'Adaboost Regressor' : AdaBoostRegressor(),
                'Bagging Regressor' : AdaBoostRegressor(),
                'Catboost Regressor' : CatBoostRegressor(),
                'XGB Regressor' : XGBRegressor()  

            }
            
            params = { # Creating a dictionary with the parameters for each Model
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K Neighbours Regressor" :{
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p' : [1, 2]
                },
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Catboost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Ridge Regression" :{
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                },
                "Lasso Regression" :{
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                },
                "ElasticNet Regression" :{
                    'alpha': [0.01, 0.1, 1.0, 10.0],  
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                "Support Vector Regressor" :{
                    'C': [0.1, 1.0, 10.0]
                },
                "Bagging Regressor" :{
                    'n_estimators': [10, 50, 100]
                },
               
            }

            model_report:dict=self.evaluate_model(X_train, y_train, X_test , y_test, models, params) # This function created inside the utils.py

            #Sorting and extracting the best model using the model score
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Nested lists concept used here

            best_model = models[best_model_name]

            logging.info("Best found model on both training and testing datasets")

            save_object(# Creating the Model.pkl file corresponding to the best model that we will get
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return (
                r2_square,
                best_model_name
            )
          
        except Exception as e:
            raise CustomException(e,sys)
          
          
    def evaluate_model(self,X_train, y_train, X_test , y_test, models, params):
        try:
            report = {}

            for i in range(len(list(models))):

                model=list(models.values())[i]
                para = params[list(models.keys())[i]] #Extracting the parameters for each model

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_) # The best parameters we get are used for the respective model
                model.fit(X_train,y_train)  # We are training the model here with the best parameters.
            

                y_train_pred = model.predict(X_train) # Predictions from the model
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train,y_train_pred) # R2 scores of the training and test datasets for the models
                test_model_score = r2_score(y_test,y_test_pred)

                report[list(models.keys())[i]] = (test_model_score,gs.best_params_) # Adding all the reports for each individual model in the report dictionary

                return report
   
        except Exception as e:
            raise CustomException(e,sys)