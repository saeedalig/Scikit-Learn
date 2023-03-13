# -------------------------------------------------------------------StandardScaler-------------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data.csv')

# Split the dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# -----------------------------------------------------------------------RobustScaler-------------------------------------------------------------------------------


import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data.csv')

# Split the dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
  
#  -----------------------------------------------------------------------MinMaxScaler-------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('data.csv')

# Split the dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# create a MinMaxScaler object


# Standardize the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 

