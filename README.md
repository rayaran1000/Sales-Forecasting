
# Sales Forecasting

![Screenshot 2024-01-08 162255](https://github.com/rayaran1000/Sales-Prediction/assets/122597408/da0afd38-2dff-432d-9928-46d7d571c748)

### Primary objective
Generate accurate predictions of future sales to facilitate effective business planning, inventory management, and resource allocation.

### Secondary objective
1. Identify and analyze key drivers influencing sales, including seasonality, promotions, economic factors, and external events, to enhance forecasting accuracy.
2. Ensure timely forecasting to provide the business with actionable insights well in advance for proactive decision-making.
3. Build robust forecasting models capable of adapting to changes in market conditions, maintaining accuracy over time.
4. Enhance interpretability of the models to provide stakeholders with insights into the factors contributing to sales predictions.
5. Implement forecasting models in a user-friendly manner to enable stakeholders with varying technical expertise to understand and utilize the results effectively.


## Directory Structure 

```plaintext
/project
│   README.md
│   requirements.txt
|   exceptions.py
|   logger.py
|   utils.py
|   application.py
|   setup.py
|   Webpage
└───artifacts
|   └───model.pkl
|   └───processor.pkl
|   └───raw.csv
|   └───test.csv
|   └───train.csv
└───logs
└───notebook
|   └───data
|       └───Sales-Forecasting.csv
|       EDA on Sales Forecasting
|       Model Training and Evaluation   
└───src
|   └───components
|       └───data_ingestion.py
|       └───data_transformation.py
|       └───model_trainer.py
|   └───pipelines
|       └───prediction_pipeline.py
|       └───training_pipeline.py
└───templates
|   └───home.html
|   └───index.html

```
## Installation

For Installing the necessery libraries required 

```bash
  pip install -r requirements.txt
```
    
## Deployment

To deploy this project run

1. To start the training pipeline 

```bash
  python src/pipelines/training_pipeline.py
```

2. Once the model is trained, to run the Flask application

```bash
  python application.py
```

3. Go to 127.0.0.1/predictdata to get the webpage

4. Use Ctrl + C in terminal to stop the server 

## Dataset Used

Kaggle Dataset - Superstore Sales Dataset Used

[Dataset Link](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

This dataset has 18 columns, namely: 

Dataset Columns:

Row ID : 'Unique Order ID for each Customer'
Order ID: 'Unique ID for each row',
Order Date: 'Order Date of the product',
Ship Date: 'Shipping Date of the Product',
Ship Mode: 'Shipping Mode specified by the Customer',
Customer ID: 'Unique ID to identify each Customer',
Customer Name: 'Name of the Customer',
Segment: 'The segment where the Customer belongs',
Country: 'Country of residence of the Customer',
City: 'City of residence of of the Customer',
State: 'State of residence of the Customer',
Postal Code: 'Postal Code of every Customer',
Region: 'Region where the Customer belong',
Product ID: 'Unique ID of the Product',
Category: 'Category of the product ordered',
Sub-Category: 'Sub-Category of the product ordered',
Product Name: 'Name of the Product',
Sales: 'Sales of the Product'
## Exploratory Data Analysis Path followed:


> 1. Importing a dataset

> 2. Understanding the big picture

> 3. Preparation / Data Cleaning

> 4. Understanding and exploring Data

> 5. Study of the relationships between variables

> 6. Plotting Data to infer results

> 7. Conclusion


## Model Training and Evaluation

Models used in the pipeline : Linear Regression , Random Forest , Decision Tree , Gradient Boosting , K Neighbours Regressor , Adaboost Regressor , Support Vector Regressor , Bagging Regressor,Ridge Regressor,Lasso Regressor,Elastic Net Regressor,CatBoost Regressor

Best Model Selected on Basis on Evaluation Metric : Random Forest Regressor

Evaluation Metric Used : R2_score


## Acknowledgements

I would like to express my gratitude to the following individuals and resources that contributed to the successful completion of this Salees Forecasting project:

- **[Kaggle]**: Special thanks to the Kaggle for providing access to the dataset and valuable insights into the industry challenges.

- **Open Source Libraries**: The project heavily relied on the contributions of the open-source community. A special mention to libraries such as scikit-learn, pandas, and matplotlib, which facilitated data analysis, model development, and visualization.

- **Online Communities**: I am grateful for the support and knowledge shared by the data science and machine learning communities on platforms like Stack Overflow, GitHub, and Reddit.

This project was a collaborative effort, and I am grateful for the support received from all these sources.


