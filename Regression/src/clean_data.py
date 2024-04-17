import os
import pandas as pd


# Load one directory back
one_back = os.path.normpath(os.getcwd() + os.sep + os.pardir)

# Load data
df = pd.read_csv(os.path.join(one_back,'housing.csv'))

df.info()
df.describe()

print(df.shape)
df.dtypes

print(df.isna().any()) # print what columns have na
init = df.shape[0]
df.dropna(inplace=True)
after = df.shape[0]
print('Lost',init - after)
print('End',after)

df_mix = df.sample(df.shape[0])
print(df_mix.shape)
# no ocean proximity, no feature engineering. total_bedrooms/total_rooms. households/population ... etc.
df_x= df_mix[['housing_median_age','total_bedrooms','total_rooms','population','households','median_income']]
df_y= df_mix['median_house_value']


# Feature Engineering

df_mix = df.sample(df.shape[0])
print(df_mix.shape)
# no ocean proximity, no feature engineering. total_bedrooms/total_rooms. households/population ... etc.
df_mix['bedrooms_per_house'] = df_mix['total_bedrooms']/df_mix['total_rooms']
df_mix['houses_available'] = df_mix['households']/df_mix['population']
# df_mix['age_income_ratio'] = df_mix['housing_median_age']/df_mix['median_income']

print(df_mix.dtypes)
df_x= df_mix[['bedrooms_per_house','houses_available','housing_median_age','median_income']]
df_y= df_mix['median_house_value']

df_mix.to_csv(os.path.join(one_back,'housing_clean.csv'),index=False)

