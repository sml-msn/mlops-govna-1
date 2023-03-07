import wget
import pandas as pd
import os

print('data creation started')

os.mkdir('data')
dataF = wget.download('''https://vk.com/doc67260180_658899570?hash=sySxNkImsY40FFyNzdvRW3wwiZgRH1fYXNz9ffF5c3T&dl=5ZW93JBfeyc625TndxLGJSuN2RCBta2c8hOgUroc1Gg''', os.path.join('data','data.csv'))
targetF = wget.download('''https://vk.com/doc67260180_658899571?hash=WVZXdOXkzpIb0KgTrtxHbn35xvoxO5ho7YVbS1OH4Fw&dl=kXvhRREF7PySTh0TCD1O6P7sNzRXFXfaZM113uKwlP8''', os.path.join('data','target.csv'))

print('\nFiles downloaded:', dataF, targetF)

data = pd.read_csv(dataF)
target = pd.read_csv(targetF)
newData = data.merge(target, how='inner', on='index')
newData.to_csv(os.path.join('data','trainData.csv'))

print('File created:', os.path.join('data','trainData.csv'))



