import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load message file and categories file into a dataframe
    
    Args:
       messages_filepath: str. message file full path
       categories_filepath: str. categories file full path
       
    Returns:
        df: DataFrame. Merged dataset of messages and categories    
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # remove duplicate rows and keep first
    messages = messages.drop_duplicates(subset=['id'], keep='first')
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # remove duplicate rows and keep first
    categories = categories.drop_duplicates(subset=['id'], keep='first')    

    # - Merge the messages and categories datasets using the common id
    df = messages.merge(categories, how='inner', on=['id'])
    
    return df

def clean_data(df):
    """Clean dataframe by splitting 'categories' column into 36 columns, and
    update values of the columns with either 0 or 1
    
    Args:
        df: DataFrame. Merged dataset of 'categories' and 'messages'
        
    Returns
        df: DataFrame. Dataframe with splitted 'categories'    
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.loc[0, :].values.tolist()    
    
    new_category_colnames = []

    # Append all of 36 individual category columns by dropping last 2 
    # chars into a list 'new_category_colnames'
    for n in category_colnames:
        new_category_colnames.append(n[:-2])    
        
    # rename the columns of `categories`
    categories.columns = new_category_colnames        
    
    for column in categories.columns:

        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Mod all column values by 2
    categories = categories.mod(2)    
    
    df = df.drop(columns=['categories'])
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """Load dataframe 'df' into sqlite database"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessageResponse', engine, if_exists='replace', index=False)
    

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