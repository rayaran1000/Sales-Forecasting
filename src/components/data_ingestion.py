import os
import sys
from src.exception import CustomException #Importing CustomException function from exception.py present inside src folder
from src.logger import logging #Importing Logging function from logging.py present inside src folder
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder

import pandas as pd 

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts','train.csv')# Training data file path
    test_data_path : str=os.path.join('artifacts','test.csv')# Testing data file path
    raw_data_path : str=os.path.join('artifacts','raw.csv')# Raw data file path

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion')
        try:

            df = pd.read_csv('notebook\data\Sales Forecast.csv') # Reading the dataset from the notebooks folder
            logging.info('Read the dataset data')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # Creating artifacts directory

            df.to_csv(self.ingestion_config.raw_data_path,index=False) # Moving entire dataset to raw.csv

            df = df.drop_duplicates()

            cat_cols = ['Segment','Region','Category','Sub-Category','Product Name']

            for columns in cat_cols:
                label_encoder = LabelEncoder()
                df[columns] = label_encoder.fit_transform(df[columns].values)

            train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)# Splitting data into train and test dataset

            train_df.to_csv(self.ingestion_config.train_data_path,index=False)# Moving training dataset to train.csv
            test_df.to_csv(self.ingestion_config.test_data_path,index=False)# Moving testing dataset to test.csv
            logging.info('Train Test Split Completed')

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_array,test_array,processor_file_path = data_transformation.initiate_data_transformation(train_path,test_path)
    model_trainer = ModelTrainer()
    r2_square,best_model_name = model_trainer.initiate_model_trainer(train_array,test_array)
    print(r2_square)
    print(best_model_name)
