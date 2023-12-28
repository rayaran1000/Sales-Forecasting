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

            unique_categories = df['Product Name'].unique()

            train_data = []
            test_data = []

            # Loop through each unique category
            for category in unique_categories:
                # Split the data for each category
                category_data = df[df['Product Name'] == category]
                if len(category_data) == 1:
                    train_category, test_category = category_data, pd.DataFrame(columns=df.columns)
                else:
                    train_category, test_category = train_test_split(category_data, test_size=0.3, random_state=42)

                # Append the split data to the training and testing sets
                train_data.append(train_category)
                test_data.append(test_category)

            # Concatenate the training and testing sets
            train_df = pd.concat(train_data, ignore_index=True)
            test_df = pd.concat(test_data, ignore_index=True)

            train_df.to_csv(self.ingestion_config.train_data_path,index=False)# Moving training dataset to train.csv
            test_df.to_csv(self.ingestion_config.test_data_path,index=False)# Moving testing dataset to test.csv

            logging.info('Train Test Split Completed')

            return(

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
