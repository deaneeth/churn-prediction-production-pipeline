import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


# Abstract Base Class for Missing Value Handling Strategies
class MissingValueHandlingStrategy(ABC):    # the class that inherited by 'Abstract base class' is the 'MissingValueHandlingStrategy'
    @abstractmethod
    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        pass
    

# Concrete implementation for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
            self.critical_columns = critical_columns
            logging.info(f"Dropping rows with missing values in critical columns: {self.critical_columns}")   # log which columns are critical
            
            
    def handle(self, df):
        df_cleaned = df.dropna(subset=self.critical_columns)
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f" {n_dropped} has been dropped")   # log how many rows are dropped
          

# Custom Imputer using LLM (Groq)
class Gender(str, Enum):
    MALE = 'Male'
    FEMALE = 'Female'
    
class GenderPrediction(BaseModel):
    firstname: str
    lastname: str
    pred_gender: Gender


# Gender Imputer Class 
class GenderImputer:
    def __init__(self):
         self.groq_client = groq.Groq()
         
    def _predict_gender(self, firstname, lastname): # predict the gender based on the firstname and lastname
        propmt = f"""
                    What is the most likely gender (Male or Female) for someone with the first name '{firstname}' and last name '{lastname}' ?
                    
                    Your response only consists of one word: Male or Female
                    """
                    
        response = self.groq_client.chat.completions.create(
                                                                model='llama-3.3-70b-versatile',
                                                                messages=[{
                                                                    "role": "user",
                                                                    "content": propmt
                                                                }],
                                                            )
        predicted_gender = response.choices[0].message.content.strip()
        prediction = GenderPrediction(firstname=firstname, lastname=lastname, pred_gender=predicted_gender)
        logging.info(f"Predicted Gender for {firstname} {lastname}: {prediction}")
        return prediction.pred_gender
    
    
    def impute(self, df):
        missing_gender_index = df['Gender'].isnull()
        
        for idx in df[missing_gender_index].index:
            firstname = df.loc[idx, 'Firstname']
            lastname = df.loc[idx, 'Lastname']
            gender = self._predict_gender(firstname, lastname)
            
            if gender:
                df.loc[idx, 'Gender'] = gender
                print(f"{firstname} {lastname} : {gender}")
                
            else:
                print(f"{firstname} {lastname} : No Gender Detected.")
                
        return df
    

# Concrete implementation for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    
    """
    
    Missing -> Mean (Age)         # for numerical column like age, we can use mean or median to fill the missing values
    Missing -> Custom (Gender)    # use LLM to predict the gender based on the firstname and lastname 
      
    """
    
    def __init__(
                    self, 
                    method='mean', 
                    fill_value=None, 
                    relevant_column=None, 
                    is_custom_imputer=False,
                    custom_imputer=None
                    ):
    
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer

    def handle(self, df):
        if self.is_custom_imputer:
            return self.custom_imputer.impute(df)
        
        df[self.relevant_column] = df[self.relevant_column].fillna(df[self.relevant_column].mean())
        logging.info(f"Missing Values filled in column {self.relevant_column}.")
        
        return df