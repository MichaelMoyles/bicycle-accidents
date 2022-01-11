# Importing all relevant packages for the project
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Import the two .csv files, Bikers.csv and Accidents.csv to be used in the project. Read the .csv files into the
# project and convert them into DataFrames using Pandas
bikers = pd.read_csv("Bikers.csv")
accidents = pd.read_csv("Accidents.csv")

# Print out a summary of the relevant information about the two files. Included is top 5 rows of data, the types of
# objects, non-null value counts, memory usage and shape of the DataFrame
print(bikers.head(), bikers.info(), bikers.shape)
print(accidents.head(), accidents.info(), accidents.shape)

# Merge the bikers and accidents DataFrames using an outer merge to avoid any data loss
acc_biker_merge = accidents.merge(bikers, how='outer')

# Check if the Accident_Index only contains unique values for each row by comparing the shape of the drop_duplicates
# DataFrame against the acc_biker_merge DataFrame to see if they have the same number of rows and columns
drop_duplicates = acc_biker_merge.drop_duplicates(subset=['Accident_Index'])
print(acc_biker_merge.shape, drop_duplicates.shape)

# Set the index to the unique ID, 'Accidents_Index' and inspect the results
acc_biker = acc_biker_merge.set_index('Accident_Index')
print(acc_biker.head())

# The dataset documentation shows 'Unknown', 'Missing Data' and 'Missing data' are used for NaN values, below replaces
# that list of values with the standard Python NaN using numpy and checks that it was done correctly
Missing_Data = ['Unknown', 'Missing Data', 'Missing data']
acc_biker = acc_biker.replace(Missing_Data, np.nan)
acc_biker_count_a = acc_biker[acc_biker == 'Unknown'].sum()
acc_biker_count_b = acc_biker[acc_biker == 'Missing Data'].sum()
acc_biker_count_c = acc_biker[acc_biker == 'Missing data'].sum()
acc_biker_count_sum = acc_biker_count_a + acc_biker_count_b + acc_biker_count_c
print(acc_biker_count_sum)

# Counts instance of NaN value in the DataFrame
count_nan_a = acc_biker.isna().sum()
print(count_nan_a)

# Replace NaN value with Mode for Categorical Data, Median for Numerical Data. NaN values in Date are replaced using
# the bfill method to prevent the creation of an outlier through creating multiple entries on a single date
mode_list = ['Time', 'Speed_limit', 'Road_conditions', 'Weather_conditions', 'Day', 'Road_type', 'Light_conditions']
median_list = ['Number_of_Vehicles', 'Number_of_Casualties']

for column in mode_list:
    acc_biker[column].fillna(acc_biker[column].mode()[0], inplace=True)

for column in median_list:
    acc_biker[column].fillna(acc_biker[column].median(), inplace=True)

# The rows are sorted by 'Time' to quasi-distribute the NaN values within 'Date' prior to filling them and then sorting
# again by 'Date' and 'Time'
acc_biker = acc_biker.sort_values(by='Time')
acc_biker_d = acc_biker['Date'].fillna(method='bfill', inplace=True)
acc_biker = acc_biker.sort_values(by=['Date', 'Time'])

# Counting instances of NaN values in the DataFrame after cleaning the data
count_nan_b = acc_biker.isna().sum()
print(count_nan_b)

# Identifying outliers in the Number_of_Casualties column of the DataFrame. This was identified as containing outliers
# in the dataset documentation
plt.figure(figsize=(10, 6))
outliers_a = sns.boxplot(x=acc_biker['Number_of_Vehicles'], y=acc_biker['Number_of_Casualties'])
outliers_a.set_xlabel('No. of Vehicles', fontsize=12)
outliers_a.set_ylabel('Number of Casualties', fontsize=12)
outliers_a.set_title("Outlier Casualties ('79 to '18)", y=1.03, fontsize=16, fontweight='bold')
outliers_a.spines[['right', 'top']].set_visible(False)

# Print number of rows within the column Number_of_Casualties that are above 20, remove all data with a value above 20
print(acc_biker[acc_biker['Number_of_Casualties'] >= 20])
acc_biker_filter = acc_biker.drop(acc_biker[acc_biker['Number_of_Casualties'] >= 20].index)

# Plotting the boxplot again to show the removal of outliers from the DataFrame
plt.figure(figsize=(10, 6))
outliers_b = sns.boxplot(x=acc_biker_filter['Number_of_Vehicles'], y=acc_biker_filter['Number_of_Casualties'])
outliers_b.set_xlabel('Number of Vehicles', fontsize=12)
outliers_b.set_ylabel('Number of Casualties', fontsize=12)
outliers_b.set_title("Outlier Casualties Removed ('79 to '18)", y=1.03, fontsize=16, fontweight='bold')
outliers_b.spines[['right', 'top']].set_visible(False)

# Descriptive statistics using built-in Pandas functions to find the median of numerical variables and mode of
# categorical variables
avg_number_vehicles = acc_biker_filter['Number_of_Vehicles'].median()
print(avg_number_vehicles)
avg_number_casualties = acc_biker_filter['Number_of_Vehicles'].median()
print(avg_number_casualties)
avg_time = acc_biker_filter['Time'].mode()
print(avg_time)
avg_speed = acc_biker_filter['Speed_limit'].mode()
print(avg_speed)

# Comparing the mean and median of numerical columns in the DataFrame, grouped by Road_type
acc_biker_type_mean = acc_biker_filter.groupby(['Road_type']).mean()
acc_biker_type_med = acc_biker_filter.groupby(['Road_type']).median()
print(acc_biker_type_mean)
print(acc_biker_type_med)


# Custom function that checks if an input is within certain times of the day and returns the relevant section of the day
def conditions(x):
    if '06:00' < x <= '12:00':
        return 'morning'
    elif '12:00' < x <= '17:00':
        return 'noon'
    elif '17:00' < x <= '21:00':
        return 'evening'
    else:
        return 'night'


# Using numpy to vectorise the custom function, inputting the data within the column 'Time' and appending the output of
# the custom function to the relevant rows within the DataFrame
func = np.vectorize(conditions)
dataset = func(acc_biker_filter['Time'])
acc_biker_filter['DaySection'] = dataset

# Checking the new 'DaySection' column is correct and finding the median and mean of numerical values within the
# DataFrame grouped by 'DaySection'
print(acc_biker_filter[['Time', 'DaySection']])
print(acc_biker_filter.groupby('DaySection').mean())
print(acc_biker_filter.groupby('DaySection').median())

# Separating out the 'Gender' column using .loc to extract all associated rows and printing the first 5 rows of the new
# DataFrame 'gender_data'
gender_data = acc_biker_filter.loc[:, ['Gender']]
print(gender_data.head())

# Using iloc and matplotlib to show whether there is a higher number of accidents amongst men, woman or other
plt.figure(figsize=(10, 6))
ax1 = plt.subplot()
gender_data.iloc[:, 0].value_counts().plot.bar(color=['blue', 'orange', 'red'])
plt.xlabel('Gender', fontsize=12)
plt.xticks(rotation=0)
plt.ylabel('Number of Accidents', fontsize=12)
plt.title("Number of Accidents by Gender ('79 to '18)", y=1.03, fontsize=16, fontweight='bold')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Using seaborn to show in what sections of the day do the majority of accidents occur
plt.figure(figsize=(10, 6))
day_count = sns.countplot(x='DaySection', data=acc_biker_filter)
day_count.set_xlabel('Section of the Day', fontsize=12)
day_count.set_ylabel('Number of Accidents', fontsize=12)
day_count.set_title("Accidents per Section of the Day ('79 to '18)", y=1.03,  fontsize=16, fontweight='bold')
day_count.set_xticklabels(['Morning (6am to 12pm)', 'Noon (12pm to 5pm)', 'Evening (5pm to 9pm)', 'Night (9pm to 6am)'
                           ])
day_count.spines[['right', 'top']].set_visible(False)

# Creation of a new DataFrame 'date_data' with the 'Date' column and the index reset. The index is reset with
# Accident_Index to be used to count instances on particular dates. The string data is converted to datetime using
# Pandas. The year is extracted to reduce noise within the data when plotting
date_data = acc_biker_filter['Date'].reset_index()
date_data['Date'] = pd.to_datetime(date_data['Date'], format='%Y-%m-%d')
date_data['Year'] = date_data['Date'].dt.year
print(date_data)

# The data is grouped by Year and instances per year are counted
year_data = date_data.groupby('Year').count()
print(year_data)

# Using seaborn the instances per year using the unique values per row in the 'Accident_Index' column are mapped against
# the 'Year' column to show accidents per year over the period
plt.figure(figsize=(10, 6))
accidents_count = sns.lineplot(data=year_data, x='Year', y='Accident_Index')
accidents_count.set_ylabel('Number of Accidents', fontsize=12)
accidents_count.set_xlabel('Year', fontsize=12)
accidents_count.set_title("Accidents per Year ('79 to '18)", y=1.03, fontsize=16, fontweight='bold')
plt.xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2018], ['1980', '1985', '1990', '1995',
                                                                    '2000', '2005', '2010', '2015', '2018'])
accidents_count.spines[['right', 'top']].set_visible(False)

# Seaborn is used to present the number of accidents within the different age cohorts in the DataFrame
plt.figure(figsize=(10, 6))
age_cohort = sns.countplot(data=acc_biker_filter, x='Age_Grp', order=['6 to 10', '11 to 15', '16 to 20', '21 to 25',
                                                                      '26 to 35', '36 to 45', '46 to 55', '56 to 65',
                                                                      '66 to 75'])
age_cohort.set_xlabel('Age Cohort', fontsize=12)
age_cohort.set_ylabel('Number of Accidents', fontsize=12)
age_cohort.set_title("Accidents per Age Cohort  ('79 to '18)", y=1.03, fontsize=16, fontweight='bold')
age_cohort.spines[['right', 'top']].set_visible(False)

# The below gets the population of the UK from the World Bank API, it iterates through the data in json format and
# appends the list 'ind' with the given values gathered as integers
pop = requests.get(
    'http://api.worldbank.org/v2/countries/gbr/indicators/SP.POP.TOTL?format=json&per_page=200&date=1979:2018')
ind = []
for obj in pop.json()[1]:
    ind.append(int(obj['value']))

# The list 'ind' is converted into a DataFrame 'pop_data' with the column name 'Total_Pop'
pop_data = pd.DataFrame(ind, columns=['Total_Pop'])

# A new column is added to 'pop_data' containing the relevant years per row of population data. The API returns the
# indicator values from the most recent year, hence the list is created with the years in reverse order.
pop_data['Year'] = pd.DataFrame([year for year in range(2018, 1978, -1)])
print(pop_data)

# The index of the year_data DataFrame is reset prior to merging datasets
year_data = year_data.reset_index()

# The year_data and pop_data DataFrames are merged using a left merge and the result is printed to check for errors
pop = year_data.merge(pop_data, how='left')
print(pop)

# Using matplotlib, a scatter plot is created to check for correlation between the total population of the UK and the
# number of accidents per year
plt.figure(figsize=(10, 6))
ax2 = plt.subplot()
plt.scatter(pop['Total_Pop'], pop['Accident_Index'])
plt.xlabel('Total Population', fontsize=12)
plt.xticks([56000000, 58000000, 60000000, 62000000, 64000000, 66000000, 68000000], ['56m', '58m', '60m', '62m', '64m',
                                                                                    '66m', '68m'])
plt.ylabel('Number of Accidents', fontsize=12)
plt.title("Total Population vs Number of Accidents ('79 to '18)", y=1.03, fontsize=16, fontweight='bold')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()
