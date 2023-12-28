from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.components.data_ingestion import DataIngestion

from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application

#Route for home page

@app.route('/')
def index():
    return render_template('index.html') # Defining the Index Html Page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html') # Home.html will contain fields for getting out Input fields
    else: # Else means this is a post request( we will be creating a Custom class in predict pipeline which will be called here)
        data=CustomData( # Here we are getting all the Input values from the webpage
            Order_ID = request.form.get('Order ID'),
            Order_Date = request.form.get('Order Date').strip(),
            Ship_Date = request.form.get('Ship Date').strip(),
            Ship_Mode =  request.form.get('Ship Mode'),
            Customer_ID = request.form.get('Customer ID'),
            Customer_Name = request.form.get('Customer Name'),
            Segment = request.form.get('Segment'),
            Country = request.form.get('Country'),
            City = request.form.get('City'),
            State = request.form.get('State'),
            Postal_Code = request.form.get('Postal Code'),
            Region = request.form.get('Region'),
            Product_ID = request.form.get('Product ID'),
            Category = request.form.get('Category'),
            Sub_Category = request.form.get('Sub-Category'),
            Product_Name = request.form.get('Product Name')

        )
        pred_df=data.get_data_as_data_frame() # We are getting the dataframe here
        pred_copy = data.get_data_as_data_frame() #Copy created because feature engineering takes 2 dfs

        data_ingestion = DataIngestion()
        
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df,pred_copy) # Here we are sending the dataframe we created in earlier step for preprocessing and model prediction
        return render_template('home.html',results=results[0]) #Since results will be in list format
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) # Maps with 127.0.0.1