# pulling local data separate from API

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

# Reading the dataset
df = pd.read_csv("/Users/nuryavuz/Desktop/inst414-capstone/tourism_arrivals.csv")
df.head()

# Counting number of NA values dropped 
before = len(df)
df = df.dropna()
after = len(df)

before
after


