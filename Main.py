import pandas as pd

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)
