import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy as sqla

def load_data(messages_filepath, categories_filepath):
    """
    Function to load data from messages_filepath and categories_filepath files.

    Args:
    messages_filepath: path used to the `messages` dataset
    categories_filepath: path used to the `categories` dataset
    
    Returns:
    df_merged: cleaned and merged dataset
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge dataset
    df_merged = pd.merge(messages, categories, on="id", how="left")
    
    return df_merged

def clean_data(df):
    """
    Function to clean dataset 
    
    Args:
    df: dataframe to be cleaned
    
    Returns:
    df_cleaned: end product of cleaned DF
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe and have them as column names
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:] # apply lambda function
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # iterate through the categories values 
    # keep only the last character of each string (the 1 or 0)
    # convert to numeric
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove rows with related = 2 value to avoid haivng other values than 0 and 1
    df = df[df['related'] != 2]
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Saves dataframe to designated database_filename path stored in SQLite DB
    
    Args:
    df: dataframe to be saved
    database_filename: database path
    """
    
    engine = sqla.create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages_final', engine, index=False, if_exists='replace') # save to table

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()