# Disaster Response Pipeline Project

### Summary of the Project:

This project aims to help emergency departments response disasters by quickly identifying specific needs and assistances that people needs to be received via data scientce multi-classification models. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifierV6.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Description

<code>
app/
  |- data/
  | |- DisasterResponse.db  #SQLite DB file
  | |- disaster_categories.csv #Raw category data
  | |- disaster_messages.csv #Raw message data
  | |- process_date.py # script to ETL raw data into SQLite DB
  |- models/
  | |- classifierV6.pkl # model file
  | |- train_classifier.py # ML Training script
  |- templates/
  | |- go.html # Query text page 
  | |- master.html # Main page
  |- run.py # Web entry script
  |- utils.py # tokenize libs
README.md
requirements.txt # required python libs
</code>
