import pandas as pd
from sklearn.model_selection import train_test_split
import os

print('preprocessing started')

trn = pd.read_csv(r'data/trainData.csv')

cat_columns = []
num_columns = []

def CatNum(df, show = False):
  global cat_columns
  global num_columns
  cat_columns = []
  num_columns = []
  for column_name in df.columns:
      if (df[column_name].dtypes == object):
          cat_columns +=[column_name]
      else:
          num_columns +=[column_name]
  if show:
    print('Категориальные данные:\t ',cat_columns, '\n Число столблцов = ',len(cat_columns))
    print('Числовые данные:\t ',  num_columns, '\n Число столблцов = ',len(num_columns))
    print('=================\n')
    
print('drop duplicates')
trn = trn.drop_duplicates()
trn = trn.drop_duplicates(['polution'])

print('drop boundary values')
question_dist = trn[trn.polution < 10e-2] 
trn = trn.drop(question_dist.index)
question_dist = trn[trn.polution > 2.5] 
trn = trn.drop(question_dist.index)

CatNum(trn, 1)
print('drop waste columns')
waste = ['index', 'period', 'population', 'tourists', 'venue', 'rate', 'food', 'glass', 'metal', 'other', 'paper', 'plastic', 'leather', 'green_waste', 'waste_recycling']
trn = trn.drop(columns=waste)
CatNum(trn, 1)

print('''OHE columns: code, year, Country, id''')
trn = pd.get_dummies(trn, columns=['code', 'year', 'Country'], drop_first= False)
trn = pd.get_dummies(trn, columns=['id'], drop_first= False)
CatNum(trn)

df_train, df_test = train_test_split(trn.dropna(), test_size=0.2, random_state=42)

os.mkdir('train')
df_train.to_csv(os.path.join('train','df_train.csv'))

os.mkdir('test')
df_test.to_csv(os.path.join('test','df_test.csv'))


print('\nFiles saved:')
print(os.path.join('train','df_train.csv'))
print(os.path.join('test','df_test.csv'))
