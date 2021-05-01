import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import joblib
# from sklearn.externals import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import classification_report

sys.path.insert(1, '../models')
from utils import tokenize

def load_data(database_filepath):
    """Load dataset from sqlite DB to dataframe. Extract datasets and category names
    
    Args:
        database_filepath: str. sqlite DB file path
        
    Return:
        X: Input dataset
        Y: Output dataset
        category_names: names of categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessageResponse', engine)
    
    X = df['message']
    Y = df[df.columns[4:]]
    
    category_names = df.columns[4:]
    
    return X, Y, category_names

def build_model():
    """Build ML Pipeline with Grid Search"""
    pipeline = Pipeline([
        ('vect' , CountVectorizer(tokenizer=tokenize)),
        ('tfidf' , TfidfTransformer()),
        ('clf' , MultiOutputClassifier(RandomForestClassifier()))
        ])

#     parameters = {
#         'clf__estimator__n_estimators': [50, 100],
#         'clf__estimator__min_samples_split': [2, 3, 4],
#     }
    parameters = {
        'clf__estimator__min_samples_split': [2],
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model
    
def evaluate_model(model, X_test, y_test, category_names):
    """Evaluate model by applying test data and print out classification report
    
    Args:
        model: model object.
        X_test: array
        y_test: test result dataframe
        category_names: list. category names
    Returns:
        None
    """
    y_pred = model.predict(X_test)

    for i, c in enumerate(y_test):

        print("<*** Feature - '{}' ***>\n".format(category_names[i]))    
        cr_y1 = classification_report(y_test.values[:,i],y_pred[:,i])    
        print(cr_y1)    
    

def save_model(model, model_filepath):
    """Save model file to the given filepath
    
    Args
        model: scikit-learn model. The fitted model
        model_filepath: string. The file path where the model is saved to
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()