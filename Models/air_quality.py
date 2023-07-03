# -*- coding: utf-8 -*-
"""air_quality.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/abyasingh/GGH_Ideathon/blob/main/air_quality.ipynb
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import seaborn as sns

per_day = pd.read_csv('/content/air_pollution_data.csv')

print(per_day.columns)

print(per_day['Date'])

def getArray(csvfile):
    city_per_day = np.empty((0,16))
    for row in per_day:
        city_per_day = np.vstack((city_per_day, np.array(row)))
    return city_per_day

def getCities(data):
    cities = data['City'].value_counts().to_frame()
    cities = cities.sort_index().index
    for i in cities:
        print(i)

def exploreData(data):
    getCities(data)

exploreData(per_day)

def getMissingValues(data):
    missing_val = data.isnull().sum()
    missing_val_percentage = 100 * data.isnull().sum() / len(data)
    missin_values_array = pd.concat([missing_val, missing_val_percentage], axis=1)
    missin_values_array = missin_values_array.rename(columns =
                                                     {0 : 'Missing Values', 1 : '% of Total Values'})
    missin_values_array = missin_values_array[
        missin_values_array.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    return missin_values_array

def mergeColumns(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['BTX'] = data['Benzene'] + data['Toluene'] + data['Xylene']
    data.drop(['Benzene','Toluene','Xylene'], axis=1)
    data['Particulate_Matter'] = data['PM2.5'] + data['PM10']
    return data

def subsetColumns(data):
    pollutants = ['Particulate_Matter', 'NO2', 'CO','SO2', 'O3', 'BTX']
    columns =  ['Date', 'City', 'AQI', 'AQI_Bucket'] + pollutants
    data = data[columns]
    return data, pollutants

def handleMissingValues(data):
    missing_values = getMissingValues(data)
    updatedCityData = mergeColumns(data)
    updatedCityData, pollutants = subsetColumns(updatedCityData)
    return updatedCityData, pollutants

updatedCityData, newColumns = handleMissingValues(per_day)
updatedCityData

filtered_df = updatedCityData[updatedCityData['City'] == 'Lucknow']
print(filtered_df)

def visualisePollutants(udata, columns, colors=None, save_path=None):
    data = udata.copy()
    data.set_index('Date', inplace=True)

    if colors:
        axes = data[columns].plot(marker='.', linestyle='None', figsize=(15, 15), subplots=True, color=colors)
    else:
        axes = data[columns].plot(marker='.', linestyle='None', figsize=(15, 15), subplots=True)

    for ax in axes:
        ax.set_xlabel('Years')
        ax.set_ylabel('ug/m3')

    if save_path:
        plt.savefig(save_path)

    plt.show()

custom_colors = ['red', 'green', 'blue', 'brown', 'violet', 'yellow']
visualisePollutants(updatedCityData, newColumns, colors=custom_colors, save_path='lastimg.png')

def trend_plot(updatedCityData, value, save_path=None):
    data = updatedCityData.copy()
    data['Year'] = [d.year for d in data.Date]
    data['Month'] = [d.strftime('%b') for d in data.Date]
    years = data['Year'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12,3), dpi= 80)
    sns.boxplot(x='Year', y=value, data=data, ax=axes[0])
    sns.pointplot(x='Month', y=value, data=data.loc[~data.Year.isin([2015, 2020]), :])

    axes[0].set_title('Year-wise Plot', fontsize=18);
    axes[1].set_title('Month-wise Plot', fontsize=18)

    if save_path:
        plt.savefig(save_path)
    plt.show()

value='SO2'
trend_plot(updatedCityData,value, save_path='timeplot.png')

def visualiseAQI(udata, columns, save_path=None):
    data = udata.copy()
    data.set_index('Date',inplace=True)

    axes = data[columns].plot(marker='.', alpha=0.5, linestyle='None', figsize=(16, 3), subplots=True)
    for ax in axes:
        ax.set_xlabel('Years')
        ax.set_ylabel('AQI')

    if save_path:
        plt.savefig(save_path)
    plt.show()

visualiseAQI(updatedCityData, ['AQI'], save_path='AQI_scatter.png')

value='AQI'
trend_plot(updatedCityData,value, save_path='aqi_box.png')

city= ['Lucknow']
filtered_city_day = updatedCityData[updatedCityData['Date'] >= '2016-01-01']
AQI = filtered_city_day[filtered_city_day.City.isin(city)][['Date','City','AQI','AQI_Bucket']]

AQI_com = AQI.pivot(index='Date', columns='City', values='AQI')
AQI_com.fillna(method='bfill',inplace=True)

print(AQI_com)

def getColorBar(city):
    col = []
    for val in AQI_com[city]:
        if val < 50:
            col.append('royalblue')
        elif val > 50 and val < 101:
            col.append('lightskyblue')
        elif val > 100 and val < 201:
            col.append('lightsteelblue')
        elif val > 200 and val < 301:
            col.append('lightcoral')
        else:
            col.append('firebrick')
    return col

de = getColorBar('Lucknow')

colors = {'Good':'royalblue', 'Satisfactory':'lightskyblue', 'Moderate':'lightsteelblue', 'Very Poor':'lightcoral', 'Severe':'firebrick'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

f, ax = plt.subplots(1, 1, figsize=(15,3))
ax.bar(AQI_com.index, AQI_com['Lucknow'], color = de, width = 0.75)

ax.legend(handles, labels, loc='upper right')

ax.title.set_text('Lucknow')

ax.set_ylabel('AQI')
ax.set_xlabel('Years')
plt.savefig('trend.png')

print(updatedCityData)

def aq(val):
    if pd.isna(val) or np.isnan(val):
        return 'N/A'
    if val < 50:
        return 'good'
    elif val < 101:
        return 'satisfactory'
    elif val < 201:
        return 'moderate'
    elif val < 301:
        return 'poor'
    elif val < 401:
        return 'very poor'
    else:
        return 'severe'

updatedCityData['Condition'] = updatedCityData['AQI'].apply(aq)

print(updatedCityData)

print(updatedCityData.columns)
updatedCityData.loc['Date'] = pd.to_datetime(updatedCityData['Date'])

updatedCityData = updatedCityData[['City','Date','AQI','AQI_Bucket']]
updatedCityData.head()

updatedCityData['AQI'] = updatedCityData['AQI'].fillna(updatedCityData['AQI'].mean(axis=0))
updatedCityData

updatedCityData['City'] = updatedCityData['City'].fillna('')
cities=pd.unique(updatedCityData['City'])
column1= cities+'_AQI'
column2=cities+'_AQI_Bucket'
columns=[*column1,*column2]
updatedCityData.columns

final_data= pd.DataFrame(index=np.arange('2015-01-01','2020-07-02',dtype='datetime64[D]'), columns=column1)
for city,i in zip(cities, final_data.columns):
    n = len(np.array(updatedCityData[updatedCityData['City'] == city]['AQI']))
    final_data[i][-n:] = np.array(updatedCityData[updatedCityData['City']==city]['AQI'])

final_data=final_data.astype('float64')
final_data=final_data.resample(rule='MS').mean()

final_data.tail()

a = final_data['Lucknow_AQI']
a.tail()

from statsmodels.tsa.seasonal import seasonal_decompose
Lucknow_AQI = final_data['Lucknow_AQI']
result = seasonal_decompose(Lucknow_AQI, model='multiplicative')
result.plot();
Lucknow_AQI
plt.savefig('plot.png')

Lucknow_aqi=Lucknow_AQI
Lucknow_aqi.columns=['ind', 'date', 'aqi']
Lucknow_aqi.drop('ind', axis=1, inplace=True)
print(Lucknow_aqi.columns)

Lucknow_aqi=Lucknow_aqi.set_index('date')

print(Lucknow_aqi)

train=Lucknow_aqi[:-24]
test=Lucknow_aqi[-24:-12]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

print(scaled_train)

from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
X,y = generator[0]
print(f'Given: \n{X.flatten()}')
print(f'Predict: \n {y}')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator,epochs=300)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

anomaly_dates = Lucknow_aqi.index
anomaly_values = Lucknow_aqi['aqi']
import matplotlib.pyplot as plt

plt.plot(anomaly_dates, anomaly_values, color='blue', label='Original Data')

threshold = 326

for i in range(len(Lucknow_aqi['aqi'])):
    if Lucknow_aqi.values[i] > threshold:
        plt.scatter(Lucknow_aqi.index[i], Lucknow_aqi.values[i], color='red', marker='o')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Anomaly Detection')
plt.show()

true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.plot(figsize=(12,8))
plt.plot(true_predictions)

scaler.fit(Lucknow_aqi)
scaled_City_AQI=scaler.transform(Lucknow_aqi)
generator = TimeseriesGenerator(scaled_City_AQI, scaled_City_AQI, length=n_input, batch_size=1)
test_predictions = []

first_eval_batch = scaled_City_AQI[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):


    current_pred = model.predict(current_batch)[0]


    test_predictions.append(current_pred)


    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
true_predictions=true_predictions.flatten()
true_preds=pd.DataFrame(true_predictions,columns=['Forecast'])
true_preds=true_preds.set_index(pd.date_range('2020-06-01',periods=12,freq='MS'))

plt.figure(figsize=(20,8))
plt.grid(True)
plt.plot( true_preds['Forecast'])
plt.plot( Lucknow_aqi['aqi'])
plt.savefig('predictions.png')
