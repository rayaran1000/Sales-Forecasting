import os
import sys
from src.exception import CustomException #Importing CustomException function from exception.py present inside src folder
from src.logger import logging #Importing Logging function from logging.py present inside src folder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_file_path : str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_object(self):

        try:

            # Specifying the columns to transform as per different pipelines
            numerical_cols = ['Postal Code' , 'order_day','order_month','order_year','ship_day','ship_month','ship_year']# Numerical columns
            categorical_cols = ['Ship Mode','Segment','Region','Category','Sub-Category','Product Name']# Categorical Columns where ordinal encoding is applied
        
            # Pipeline for Numerical columns
            num_pipeline = Pipeline(

                steps=[
        
                    ('scaler',StandardScaler(with_mean=False))
                ]

            )

            # Pipeline for categorical columns with ordinal encoding
            cat_pipeline1=Pipeline(
    
                steps=[

                    ('ordinalencoder',OrdinalEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
    
            )

            logging.info(f"Numerical features : {numerical_cols}")
            logging.info(f"Categorical; features : {categorical_cols}")

            preprocessor = ColumnTransformer(

                [

                ('numerical columns',num_pipeline,numerical_cols),
                ('categorical columns 1',cat_pipeline1,categorical_cols)

                ]

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Starting data transformation')

            train_df.drop('Row ID',axis=1,inplace=True)
            test_df.drop('Row ID',axis=1,inplace=True)
            train_sales = train_df['Sales'] 
            test_sales = test_df['Sales']

            #Dropping the Sales column because we are using this function during prediction as well, where we dont have the Sales Column
            train_df.drop('Sales',axis=1,inplace=True)
            test_df.drop('Sales',axis=1,inplace=True)
            print(train_df.shape)
            print(train_df.columns)

            train_df,test_df = self.feature_engineering(train_df,test_df)
            print(train_df.columns)
            print(train_df.shape)

            #Outlier treatment for the target column using logarithmic transformation
            train_df['Sales_log'] = np.log(train_sales)
            test_df['Sales_log'] = np.log(test_sales)

            preprocessor_obj = self.get_transformation_object()

            target_column_name = 'Sales_log'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) # Creating the training datasets for training by dropping the target column 
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) # Creating the testing datasets for training by dropping the target column
            target_feature_test_df=test_df[target_column_name]
            print(input_feature_train_df.columns)
            print(input_feature_train_df.shape)

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df) # Now applying the transformation pipeline on the train and test datasets
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[ # We are combining arrays column wise (adding the preprocessed independent feature columns and the last column as the target column)
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # (adding the preprocessed independent feature columns and the last column as the target column)

            logging.info(f"Saved preprocessing object.")

            save_object( # Saving the Pickle file for Preprocessor, using the function save_object defined in utils.py

                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_obj

            )

            return ( # Returning the train and test arrays with all the columns and the preprocessed columns along with Preprocessor Pickle File Path
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
    
    def feature_engineering(self,train_df,test_df):

        try:

            #Filling the missing values of Postal Code with the manually found pincode on basis on City and State
            train_df['Postal Code'].fillna('5401',inplace=True)
            test_df['Postal Code'].fillna('5401',inplace=True)

            #Converting the Postal Code column to integer
            train_df['Postal Code'] = train_df['Postal Code'].astype(int)
            test_df['Postal Code'] = test_df['Postal Code'].astype(int)

            #Dropping unnecessary columns not needed for prediction
            unnecessery_columns = ['Customer ID' , 'Order ID' , 'Product ID' , 'Customer Name' , 'Country' , 'City' , 'State']

            train_df.drop(unnecessery_columns,axis=1,inplace=True)
            test_df.drop(unnecessery_columns,axis=1,inplace=True)

            #Converting the Date-time representation columns to seperate columns
            train_df['Order Date'] = pd.to_datetime(train_df['Order Date'], format='%d/%m/%Y') 
            train_df['Ship Date'] = pd.to_datetime(train_df['Order Date'], format='%d/%m/%Y')
            test_df['Order Date'] = pd.to_datetime(test_df['Ship Date'], format='%d/%m/%Y')
            test_df['Ship Date'] = pd.to_datetime(test_df['Ship Date'], format='%d/%m/%Y')

            train_df.insert(loc=6, column='order_day', value=train_df['Order Date'].dt.day)
            train_df.insert(loc=7, column='order_month', value=train_df['Order Date'].dt.month)
            train_df.insert(loc=8, column='order_year', value=train_df['Order Date'].dt.year)

            train_df.insert(loc=9, column='ship_day', value=train_df['Ship Date'].dt.day)
            train_df.insert(loc=10, column='ship_month', value=train_df['Ship Date'].dt.month)
            train_df.insert(loc=11, column='ship_year', value=train_df['Ship Date'].dt.year)

            test_df.insert(loc=6, column='order_day', value=test_df['Order Date'].dt.day)
            test_df.insert(loc=7, column='order_month', value=test_df['Order Date'].dt.month)
            test_df.insert(loc=8, column='order_year', value=test_df['Order Date'].dt.year)

            test_df.insert(loc=9, column='ship_day', value=test_df['Ship Date'].dt.day)
            test_df.insert(loc=10, column='ship_month', value=test_df['Ship Date'].dt.month)
            test_df.insert(loc=11, column='ship_year', value=test_df['Ship Date'].dt.year)

            #Dropping the original columns because they are no longer needed
            date_cols = ['Order Date', 'Ship Date']

            train_df.drop(columns=date_cols,inplace=True)
            test_df.drop(columns=date_cols,inplace=True)

            return (
                train_df,
                test_df
            )

        except Exception as e:
            raise CustomException(e,sys)     

    
