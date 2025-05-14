import pandas as pd
from ucimlrepo import fetch_ucirepo

def demographic_data_analyzer():
    # Fetch dataset
    census_income = fetch_ucirepo(id=20)
    
    # Data (as pandas dataframes)
    df = census_income.data.features
    df['salary'] = census_income.data.targets
    
    # Metadata and variable information
    print(census_income.metadata)
    print(census_income.variables)
    
    # How many people of each race are represented in this dataset?
    race_count = df['race'].value_counts()
    
    # What is the average age of men?
    average_age_men = round(df[df['sex'] == 'Male']['age'].mean(), 1)
    
    # What is the percentage of people who have a Bachelor's degree?
    percentage_bachelors = round((df['education'] == 'Bachelors').mean() * 100, 1)
    
    # Advanced education includes Bachelors, Masters, or Doctorate
    higher_education = df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])
    lower_education = ~higher_education
    
    # Percentage of people with advanced education who earn >50K
    higher_education_rich = round((df[higher_education]['salary'] == '>50K').mean() * 100, 1)
    
    # Percentage of people without advanced education who earn >50K
    lower_education_rich = round((df[lower_education]['salary'] == '>50K').mean() * 100, 1)
    
    # What is the minimum number of hours a person works per week?
    min_work_hours = df['hours-per-week'].min()
    
    # Percentage of people who work the minimum number of hours per week and earn >50K
    min_workers = df[df['hours-per-week'] == min_work_hours]
    rich_percentage = round((min_workers['salary'] == '>50K').mean() * 100, 1)
    
    # Country with the highest percentage of people that earn >50K
    country_salary_counts = df.groupby('native-country')['salary'].value_counts(normalize=True).unstack()
    highest_earning_country = country_salary_counts[">50K"].idxmax()
    highest_earning_country_percentage = round(country_salary_counts[">50K"].max() * 100, 1)
    
    # Most popular occupation for those who earn >50K in India
    top_IN_occupation = df[(df['native-country'] == 'India') & (df['salary'] == '>50K')]['occupation'].value_counts().idxmax()
    
    # Returning results as a dictionary
    return {
        'race_count': race_count,
        'average_age_men': average_age_men,
        'percentage_bachelors': percentage_bachelors,
        'higher_education_rich': higher_education_rich,
        'lower_education_rich': lower_education_rich,
        'min_work_hours': min_work_hours,
        'rich_percentage': rich_percentage,
        'highest_earning_country': highest_earning_country,
        'highest_earning_country_percentage': highest_earning_country_percentage,
        'top_IN_occupation': top_IN_occupation
    }

# Running the function
if __name__ == "__main__":
    results = demographic_data_analyzer()
    for key, value in results.items():
        print(f"{key}: {value}")