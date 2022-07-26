"""
This is another example of a Decision Tree model and prediction using scikit learn library.

Here we predict salary (depedendent variable) by giving various types of criteria (different forks in a decision tree).
e.g. What's the salary of a particular gender, of some age, with specific education, in a certain occupation.

see blog for full explanation: http://flyingsalmon.net/?p=4070

"""
import pandas as pd

df = pd.read_csv("salary.csv")
print(df.shape) # (rows, cols): (32561, 11)
print(df.dtypes)
"""
out:
age                int64
sector            object
education         object
education-num      int64
marital-status    object
occupation        object
race              object
sex               object
hours-per-week     int64
country           object
salary            object

So, we have age and hours-per-week as the integers and rest of them are string objects meaning categorical values or labels.

"""

# Check if there's any missing values
print(df.isna().sum())
"""
out:
age               0
sector            0
education         0
education-num     0
marital-status    0
occupation        0
race              0
sex               0
hours-per-week    0
country           0
salary            0
dtype: int64

We see there's no missing data in any column.

"""
# We drop columns from dataframe that we won't use
df=df.drop(columns=['sector', 'education-num', 'marital-status', 'hours-per-week'])
print(df.shape) # out: (32561, 7) [as rows, cols]
print(df.columns) # out: Index(['age', 'education', 'occupation', 'race', 'sex', 'country', 'salary'], dtype='object')

"""
But there's a problem!
Opening the csv file in Excel, we also see there are some '?' in various columns! But it's not just '?' there's a preceding SPACE char before '?'
 So, it's " ?" that we need to remove all rows that have this string in any of its remaining 7 columns.
 
"""
import numpy as np
df=df.replace(" ?",np.nan).dropna(axis = 0, how = 'any') # replaces all cells with null/NaN that has " ?" in them.
# Then we can drop the null values as usual, effectively, removing all rows that had " ?" value in any column
df.dropna(axis = 0, how = 'any', inplace = True)
print(df.shape) # out: (30162, 7)...so, 2399 rows dropped from the dataframe df that had at least one " ?" in them.

# We could do the encoding as part of sklearn modeling or we can also do it ourselves before training in the dataset ourselves as shown below:
cleanup_nums = {"salary": {"<=70K": 0, ">70K": 1}} # NOTE: for target value such as salary, we want 0 or 1 only
                                                   # (not 1,2,3...n because some ML may put different weights on higher numbers!)
df = df.replace(cleanup_nums) 

# column 'sex' also only has "Female" and "Male", so we can convert manually the same way as above to 0 or 1
cleanup_nums = {"sex": {"Male": 0, "Female": 1}} 
df = df.replace(cleanup_nums) 

# Now our dataset is ready for ML modeling
# Save target column (to predict: salary) in a separate dataframe
target=df['salary']

# And drop the target column from the df dataframe which should only contain input values (all except target values: salary)
inputs = df.drop('salary',axis='columns')

# Next we encode all remaining non-numeric columns: education, occupation, race, country
# NOTE The following 2 lines to suppress annoying error/warning in shell that comes from sklearn lib (fix source: stackoverflow)
# These warnings suppression may not be needed if using Jupyter notebook.
import warnings 
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder # import the LabelEncoder module
education_enc = LabelEncoder() # create an encoder object
inputs['education_n'] = education_enc.fit_transform(inputs['education'])
occupation_enc = LabelEncoder() 
inputs['occupation_n'] = occupation_enc.fit_transform(inputs['occupation'])
race_enc = LabelEncoder() 
inputs['race_n'] = occupation_enc.fit_transform(inputs['race'])
country_enc = LabelEncoder() 
inputs['country_n'] = country_enc.fit_transform(inputs['country'])

# see blog for explanation: http://flyingsalmon.net/?p=4070
inputs.to_excel('salary_encoded.xlsx', index=False) 

# Next, drop the original 4 columns with labels as we don't need them anymore for ML
inputs = inputs.drop(['education','occupation','race','country'], axis='columns')
print("inputs dataframe tail:\n", inputs.tail()) # we only have these columns in inputs now: age  sex  education_n  occupation_n  race_n  country_n (values 0...n)
print("\ntarget dataframe tail:\n", target.tail()) # target has only 0 or 1 values

### Now we'll use Decision Tree alogirthm to train and then predict salary given different features
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs, target) 

print(model.score(inputs,target)) # out: 0.8660566275445926 so, 86.6% prediction accuracy.

# Now we can predict!
# PREDICTION QUERY 1: In United States, is salary of an Adm-clerical professional of female gender of race Asian-Pac-Islander with Bachelors degree > 70K?
# To set up the query, we need to get the numeric codes for each of these params (features) from our encoded columns
# To make it easy, we saved the encoded and their labels above as 'salary_encoded.xlsx' file...open it and find/filter their numeric codes.
# United States = 37 [country_n]
# Adm-clerical = 2 [occupation_n]
# Female = 1 (we already did this manually before modeling) [sex]
# Asian-Pac-Islander = 2 [race_n]
# Bachelors = 9 [education_n]
# ALSO PAY ATTENTION TO THE PARAMETERS ODER in predict() THEY MUST MATCH THE ORDER OF encoded COLUMNS ORDER in the dataframe used: inputs
# The order of columns in inputs dataframe is:  age  sex  education_n  occupation_n  race_n  country_n
# If we don't need a specific parameter for a given query set it to 0 or -1

answer=model.predict([[-1,1,9,2,2,37]])
print("USA, Admin, Female, Asian, BSc.:", answer) # out: [0] meaning: No!

# see blog for explanation and meaning of the prediction outputs and how they match up with dataset: http://flyingsalmon.net/?p=4070

# PREDICTION QUERY 2: In USA, is salary of an Exec-managerial professional of a white male gender of any race with Masters degree > 70K?
# From 'salary_encoded.xlsx' we find the following codes:
# United States = 37 [country_n]
# Exec-managerial = 4 [occupation_n]
# male = 0 (we already did this manually before modeling) [sex]
# white = 4 [race_n]
# Masters = 12 [education_n]

answer=model.predict([[-1,0,12,4,4,37]]) 
print("USA, Exec, male, Masters:", answer) # out: [1] meaning: Yes!


# PREDICTION QUERY 3: In Ireland, is salary of a 50 year professional (any job) with Some-college > 70K? (any race, gender, sex)
# From 'salary_encoded.xlsx' we find the following codes:
# Ireland = 19 [country_n]
# Age = 50 [age]
# Some-college = 15 [education_n]

answer=model.predict([[50,-1,15,-1,-1,19]]) 
print("Ireland, any job, Some-college:", answer) # out: [1] meaning: Yes!


### end