import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()

    def predict(self,features,features1):
        try:
            model_path=os.path.join("artifacts","model.pkl") 
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl') #Will handle the preprocessing part
            print("Before Loading")
            model=load_object(file_path=model_path) #Loads the Model from the Pickle file
            preprocessor=load_object(file_path=preprocessor_path) #Loads the Preprocessor from Pickle file
            print("After Loading")
            print(features.shape)
            features_engineered,features_engineered_copy = self.data_transformation.feature_engineering(features,features1)
            columns = ['Ship Mode', 'Segment', 'Postal Code', 'Region', 'order_day','order_month', 'order_year', 'ship_day', 'ship_month', 'ship_year','Category', 'Sub-Category', 'Product Name']
            feature_engineered_df = pd.DataFrame(features_engineered,columns=columns)
            print(features_engineered.shape)
            data_scaled=preprocessor.transform(features_engineered)
            preds=model.predict(data_scaled)
            preds_actual = np.exp(preds)
            return preds_actual
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData: # Class responsible for mapping all the inputs that we are getting in the HTML webpage with the backend
    def __init__(self,        
        Order_ID: str,
        Order_Date:str,
        Ship_Date:str,
        Ship_Mode:str,
        Customer_ID:str,
        Customer_Name:str,
        Segment:str,
        Country:str,
        City:str,
        State:str,
        Postal_Code:str,
        Region:str,
        Product_ID:str,
        Category:str,
        Sub_Category:str,
        Product_Name:str
        ):

#Assigning these values(coming from web application)
        self.Order_ID = Order_ID

        self.Order_Date = Order_Date

        self.Ship_Date = Ship_Date

        self.Ship_Mode = Ship_Mode

        self.Customer_ID = Customer_ID

        self.Customer_Name = Customer_Name

        self.Segment = Segment

        self.Country = Country

        self.City = City

        self.State = State

        self.Postal_Code = Postal_Code

        self.Region = Region

        self.Product_ID = Product_ID

        self.Category = Category

        self.Sub_Category = Sub_Category

        self.Product_Name = Product_Name

    def get_data_as_data_frame(self): #Returns all our input data as dataframe, because we train our models using dataframes
        try:
            custom_data_input_dict = {

                "Order ID": [self.Order_ID],
                "Order Date": [self.Order_Date],
                "Ship Date": [self.Ship_Date],
                "Ship Mode": [self.Ship_Mode],
                "Customer ID": [self.Customer_ID],
                "Customer Name": [self.Customer_Name],
                "Segment": [self.Segment],
                "Country": [self.Country],
                "City": [self.City],
                "State": [self.State],
                "Postal Code": [self.Postal_Code],
                "Region": [self.Region],
                "Product ID": [self.Product_ID],
                "Category": [self.Category],
                "Sub-Category": [self.Sub_Category],
                "Product Name": [self.Product_Name]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)