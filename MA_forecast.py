

import numpy as np
import pandas as pd

        
def MA_forecast(data,date_column='date',data_column='lineitem',forecast_points=3,window_size=6,rule='W'):
    
    data['date']=pd.to_datetime(data[date_column])
    data[data_column] = data[data_column].fillna(value=0)
    df_temp =pd.DataFrame( pd.date_range(str(min(df[date_column]).date()), periods=len(data)+forecast_points, freq=rule),columns=[date_column])
    num = data[[data_column]].to_numpy()
    for i in range(forecast_points):
        num=np.append(num,np.round(sum(num[-window_size:])/window_size,2))
    
    df_temp['Moving Average'] = num

    return df_temp 
    
    






























###########################################
https://stackoverflow.com/a/38276069
c = [4,5,6]

args = [a,b,c]

for combination in itertools.product(*args):
    print (combination)
    
    
for i in loop:
    d[i+'_unique'] = df[i].unique().tolist()
    
a = d['location_unique']
b = d['customer_segment_ID_unique']

num = {'physics': 80, 'math': 90, 'chemistry': 86}
print(list(num)[1])

mask = ''
for i in range(len(d)):
    mask = mask+f"([(df2[loop_elements[{i}]]=="+combination[i])
#######################################
d = {}
for i in loop:
    d[i+'_unique'] = df[i].unique().tolist()

args=[]
for key, value in d.items():
    print(value)
    args.append(value)
temp=[]
for combination in itertools.product(*args):
    temp.append(combination)

mask = ''
for i in range(len(d)):
    for j in range(len(temp)):
     mask = mask+f"([(df2[loop_elements[{i}]]=="+str(temp[j][i])+"&"
     

for j in range(len(temp)):
    mask = ''
    for i in range(len(d)):
        
        mask = mask+f"(df2[loop_elements[{i}]]=="+str(temp[j][i])+")"+"&"
        mask = '(' + mask[:-1] + ')'
        
df_temp=pd.DataFrame()
for j in range(1):
    for i in range((2)):
        df2 = df2.loc[df2[loop[i]]==temp[j][i]]

df_temp=pd.DataFrame()
for j in range(len(temp)):
    df2 = df.copy()
    for i in range((len(d))):
        df2 = df2.loc[df2[loop[i]]==temp[j][i]]
    df_temp.append(df2)