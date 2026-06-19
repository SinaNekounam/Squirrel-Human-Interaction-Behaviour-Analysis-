import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder


# Data preprocessing

# READ DATA
squirrel = pd.read_csv('squirrel.csv')
hectare = pd.read_csv('hectare.csv')
stories = pd.read_csv('stories.csv')

# FEATURE ENGINEERING 
# combining similar features in order to find pattern easily

# FEATURE ENGINEERING 1
# combine with physical features
# range 0~4 shows the number of behaviours observed
active_features = ['Running', 'Chasing', 'Climbing']
squirrel['active_score'] = squirrel[active_features].astype(int).sum(axis='columns')
# True/False -> integer and sum it

# FEATURE ENGINEERING 2
# combine with surviving features
# range 0~2 shows the number of behaviours observedv
foraging_features = ['Eating', 'Foraging']
squirrel['foraging_score'] = squirrel[foraging_features].astype(int).sum(axis='columns')
# True/False -> integer and sum it

# FEATURE ENGINEERING 3
# combine with sound and voice features
# range 0~3 shows the number of behaviours observed
vocal_features = ['Kuks', 'Quaas', 'Moans']
squirrel['vocal_score'] = squirrel[vocal_features].astype(int).sum(axis='columns')
# True/False -> integer and sum it

# FEATURE ENGINEERING 4
# combine with alert sign features
# range 0~2 shows the number of behaviours observed
alert_features = ['Tail flags', 'Tail twitches']
squirrel['alert_score'] = squirrel[alert_features].astype(int).sum(axis='columns')
# True/False -> integer and sum it

# Approaches, Runs from, Indifferent are excluded from features
# they will be used as the target variable later on(human_reaction)

# FEATURE ENGINEERING: time observation
# AM=0, PM=1
squirrel['shift_enc'] = (squirrel['Shift'] == 'PM').astype(int)
# Encoding PM= True, True= 1

# Feature Engineering: Date => weekday
# Range 0~6 (0=Monday, 6=Sunday)
def get_weekday(date):
    match = re.search(r'(\d{2})(\d{2})(\d{4})', str(date))
    # Split day,month,year
    if match:
        month, day, year = match.groups()
        return datetime(int(year), int(month), int(day)).weekday()
        # Change day to integer
    return None

squirrel['weekday'] = squirrel['Date'].apply(get_weekday)
# Add this to squirrel data.

# DATA INTEGRATION AND MERGING

# Apply ordinal encoding to Hectare Conditions
# Calm=0, Moderate=1, Busy=2
#Convert value which is typo or weird.
# Unknown values are set to NaN for later imputation
hectare['Hectare Conditions'] = hectare['Hectare Conditions'].replace({
    'Calm, Busy': 'Moderate',
    'Medium': 'Moderate'})
ordinalencoding_conditions = OrdinalEncoder(
    categories=[['Calm', 'Moderate', 'Busy']], handle_unknown = 'use_encoded_value',unknown_value=np.nan)

hectare['conditions_enc'] = ordinalencoding_conditions.fit_transform(hectare[['Hectare Conditions']])

# Identify threat animals that might influence squirrel behaviour
# 1 = threat animal, 0 = no threat animal
def check_threat(other_animals):
    if pd.isna(other_animals): #Check empty data
        return 0
    elif 'Dogs' in other_animals:
        return 1
    elif 'Cats' in other_animals:
        return 1
    else:
        return 0

hectare['squirrel_threat'] = hectare['Other Animal Sightings'].apply(check_threat)

# Aggregate area-level features using groupby
hectare_grouped = hectare.groupby(['Hectare', 'Shift', 'Date']).agg(
    total_squirrels=('Number of Squirrels', 'sum'),
    conditions_enc=('conditions_enc', 'mean'),
    has_threat=('squirrel_threat', 'max')).reset_index() # Organize the columns

# Merge squirrel.csv with hectare.csv
# how='left' keeps all squirrel records even if no matching hectare data
merged = pd.merge(squirrel, hectare_grouped, on=['Hectare', 'Shift', 'Date'], how='left')

# OUTLIER DETECTION & HANDLING
# to check if within Central Park boundaries

print("\n---------------------------------------------------------------------------")
print("OUTLIER DETECTION & HANDLING")
coord_outliers = merged[(merged['X'] < -73.992919921875) | (merged['X'] > -73.948974609375) |(merged['Y'] < 40.75557964275589) | (merged['Y'] > 40.805493843894155)]
if len(coord_outliers) == 0:
    print('All coordinates are in boundaries')



# Squirrel count outliers using IQR method ---Q1---Median---Q3---
Q1 = merged['total_squirrels'].quantile(0.25) #4.5
Q3 = merged['total_squirrels'].quantile(0.75) #12.0
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
merged['outlier_variable'] = (merged['total_squirrels'] > upper_limit).astype(int)
print(f'Squirrel count outliers: {merged["outlier_variable"].sum()} (marked in outlier_variable column)')


# HANDLING MISSING VALUES

# Age => mode (cannot use mean as it is categorical)
merged['Age'] = merged['Age'].fillna(merged['Age'].mode()[0])

# Primary Fur Color => mode
merged['Primary Fur Color'] = merged['Primary Fur Color'].fillna(merged['Primary Fur Color'].mode()[0])

# Highlight Fur Color => drop
# Too many missing values (36%)
merged = merged.drop(columns=['Highlight Fur Color'])

# Location => mode
merged['Location'] = merged['Location'].fillna(merged['Location'].mode()[0])

# total_squirrels, has_threat => 0
merged['total_squirrels'] = merged['total_squirrels'].fillna(0)
merged['has_threat'] = merged['has_threat'].fillna(0)

# conditions_enc => mode
merged['conditions_enc'] = merged['conditions_enc'].fillna(merged['conditions_enc'].mode()[0])

print("\n---------------------------------------------------------------------------")
print('Missing values remaining:', merged.isnull().sum().sum())

# DISCRETISATION
x_min = -73.992919921875
x_max = -73.948974609375
y_min = 40.75557964275589
y_max = 40.805493843894155

x_mid = (x_min + x_max) / 2
y_mid = (y_min + y_max) / 2

def get_zone(row):
    if row['Y'] >= y_mid and row['X'] >= x_mid:
        return 'NorthEast'
    elif row['Y'] >= y_mid and row['X'] < x_mid:
        return 'NorthWest'
    elif row['Y'] < y_mid and row['X'] >= x_mid:
        return 'SouthEast'
    else:
        return 'SouthWest'


merged['zone'] = merged.apply(get_zone, axis='columns')

print("\n---------------------------------------------------------------------------")
print('Zone distribution:')
print(merged['zone'].value_counts())


# Squirrel count => density level (Low/Medium/High)
def get_density(count):
    if count <= 4: #IQR Q1
        return 'Low'
    elif count <= 12: # IQR Q3
        return 'Medium'
    else:
        return 'High'

merged['density_level'] = merged['total_squirrels'].apply(get_density)

print("\n---------------------------------------------------------------------------")
print('Zone distribution:', merged['zone'].value_counts().to_dict())
print('Density distribution:', merged['density_level'].value_counts().to_dict())

#TARGET VARIABLE

# Create human_reaction from Approaches, Runs from, Indifferent
# 0 = runs away, 1 = indifferent, 2 = approaches
def get_reaction(row):
    if row['Approaches'] == True:
        return 2
    elif row['Runs from'] == True:
        return 0
    else:
        return 1

merged['human_reaction'] = merged.apply(get_reaction, axis='columns')

# Remove original columns after creating new target variable
merged = merged.drop(columns=['Approaches', 'Runs from', 'Indifferent'])

print("\n---------------------------------------------------------------------------")
print('Target variable distribution:')
print(merged['human_reaction'].value_counts())
print('0: Runs away, 1: Indifferent, 2: Approaches')

#Lastly make a preprocessing data to new csv
merged.to_csv('task_1_preprocessed_squirrel_data.csv',index = False)
# print('task_1_preprocessed_squirrel_data.csv')

check = pd.read_csv('task_1_preprocessed_squirrel_data.csv')
#print(check.head())

print("\n---------------------------------------------------------------------------")
print('task_1_preprocessed_squirrel_data.csv was saved')
