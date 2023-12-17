import pandas as pd


# let's read the dataset

df = pd.read_csv('C:/Users/utente/Desktop/DATA SCIENCE/ADM/HW4/vodclickstream_uk_movies_03.csv')

######################
#                    #
#       PART 1       #
#                    #
######################

# creating a new df to see which one is the most clicked title

df2 = pd.DataFrame()
df2['title'] = df['title'].unique()
df2['duration'] = [0.0] * df2.shape[0]


       
grouped_titles = df[df['duration'] >= 0].groupby('title')['duration'].sum()
df2['duration'] = df2['title'].map(grouped_titles).fillna(0).astype(float)
    
#sorting dataset by 'duration' values
top_titles = df2.sort_values(by='duration', ascending=False) 
print('\n\n - The most watched Netflix title is \"', top_titles.iloc[0]['title'], '\".', sep = '')



######################
#                    #
#       PART 2       #
#                    #
######################


# datetime was originally provided as a string, now we convert it in a proper datetime type variable

df['datetime'] = pd.to_datetime(df['datetime'])

df_by_datetime = df.sort_values(by='datetime')   #creating the dataset values with sorted datetimes
total_time = df_by_datetime.iloc[-1]['datetime'] - df_by_datetime.iloc[0]['datetime']
total_time_in_seconds =total_time.total_seconds()
avg = total_time_in_seconds/df.shape[0]   #time between first and last date registered devided by num. of clicks
print(' - The average time between subsequent clicks in Netflix is', round(avg, 3), 'seconds.')



######################
#                    #
#       PART 3       #
#                    #
######################

# creating a new df to see the most active user

df3 = pd.DataFrame()
df3['user_id'] = df['user_id'].unique()
df3['duration'] = [0.0] * df3.shape[0]

grouped_users = df[df['duration'] >= 0].groupby('user_id')['duration'].sum()
df3['duration'] = df3['user_id'].map(grouped_users).fillna(0).astype(float)

top_users = df3.sort_values(by='duration', ascending=False)
print(' - The user who spent the most time on Netflix is user n°', top_users.iloc[0]['user_id'], '\n')




