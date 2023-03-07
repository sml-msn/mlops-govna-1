import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.linear_model import Ridge
from joblib import dump, load

print('model preparation started')

trn = pd.read_csv(r'train/df_train.csv')

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

CatNum(trn, 0)
df_num = trn[num_columns].copy()

X,y = df_num.drop(columns = ['polution']).values,df_num['polution'].values

scaler  = MinMaxScaler()
scaler.fit_transform(X)
X = scaler.transform(X) 

print('model training')
model = Ridge()
model.fit(X, y)

os.mkdir('model')
dump(model, os.path.join('model','model.joblib')) 

print('\nFiles saved:')
print(os.path.join('model','model.joblib'))
