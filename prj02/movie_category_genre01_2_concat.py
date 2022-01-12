import pandas as pd
import glob

data_paths = glob.glob('./crawling/movie_genre_*.csv')

df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df_temp.columns = ['summary', 'genre']
    df = pd.concat([df, df_temp])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
print(df.tail())
print(df['genre'].value_counts())
print(df.info())
df.to_csv('./crawling/movie_genre.csv', index=False)