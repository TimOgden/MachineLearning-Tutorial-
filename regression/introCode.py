import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Since this is just a linear regression, we need to change these features into something that the algorithm can better understand
#For example, a percentage of High/Low price would be a better feature than the two separate features.

#Using formula for percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

#Now let's overwrite our dataframe with a new dataframe of only what we care about
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())