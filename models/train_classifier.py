import sys
import pandas as pd
import sqlite3
import sqlalchemy as sqla
import pickle

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Loads data from SQLite database
    
    Args:
    database_filepath: String. table path from DB
    
    Returns:
    X: Data features
    Y: Data target
    category_names: labels name
    """
    
    # loads data
    engine = sqla.create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('disaster_messages_final', con=engine)
    
    # Get messages for feature
    X = df['message']
    
    # get categories
    y = df.iloc[:, 4:]

    # get labels name
    categories_name = y.columns.values

    return X,y,categories_name


def tokenize(text):
    """
    
    Tokenize text

    Args:
    text: String. text to be tokenized
    
    Returns:
    tokens: cleaned tokens
    """
    
    # Normalize text
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lematize the tokens ; strip to trim whitespaces
    tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in tokens]
    
    return tokens


def build_model():
    """
    Builds model with pipeline
    
    Returns:
    cv: model with grid search
    """
    
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [100],
        'clf__estimator__min_samples_split' : [2]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Evaluates the model with test set and display classification report
        
    Args:
    model: trained model
    X_test: predict set
    Y_test: labeled test set
    category_names: names of the tested categories
    """
    y_pred = model.predict(X_test)
    
    for i, column in enumerate(category_names):
        print(column, classification_report(Y_test[column], y_pred[:, i]))


def save_model(model, model_filepath):
   """
   Exports model in pickle format
   
   Args:
   model: model to be formatted
   model_filepath: path to save
   """
    
   pickle.dump(model, open(model_filepath, 'wb'))

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