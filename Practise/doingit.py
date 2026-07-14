import pandas as pd


df = pd.read_excel('Sample1.xlsx')
df[Parent_Education]= df[Parent_Education].replace('?','Graduate')

print(df)
