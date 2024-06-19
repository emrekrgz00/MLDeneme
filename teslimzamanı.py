import pandas as pd
import numpy as np
import plotly.express as px

data = pd.read_csv("../data/teslimtime.txt")
# print(data.head())
# print(data.isnull().sum())

# Set the earth's radius (in kilometers)
R = 6371

# Convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Function to calculate the distance between two points using the haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
  
# Calculate the distance between each pair of points
data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])
    

# figure = px.scatter(data_frame = data, 
#                     x="distance",
#                     y="Time_taken(min)", 
#                     size="Time_taken(min)", 
#                     trendline="ols", 
#                     title = "Relationship Between Distance and Time Taken")
# figure.show()

# figure = px.scatter(data_frame = data, 
#                     x="Delivery_person_Age",
#                     y="Time_taken(min)", 
#                     size="Time_taken(min)", 
#                     color = "distance",
#                     trendline="ols", 
#                     title = "Relationship Between Time Taken and Age")
# figure.show()

# figure = px.scatter(data_frame = data, 
#                     x="Delivery_person_Ratings",
#                     y="Time_taken(min)", 
#                     size="Time_taken(min)", 
#                     color = "distance",
#                     trendline="ols", 
#                     title = "Relationship Between Time Taken and Ratings")
# figure.show()

# fig = px.box(data, 
#              x="Type_of_vehicle",
#              y="Time_taken(min)", 
#              color="Type_of_order")
# fig.show()

#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "distance"]])
y = np.array(data[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

# training the model
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(xtrain, ytrain, batch_size=1, epochs=9)

"""
optimizer (optimize edici), bir yapay sinir ağı modelinin eğitim sürecinde kullanılan bir optimizasyon algoritmasıdır.
 Modelin kaybını (loss) minimize etmek için kullanılır ve ağı güncellemek için geriye yayılım (backpropagation) sırasında gradyan iniş 
 (gradient descent) veya onun türevleri gibi teknikler kullanır. Bu, ağın eğitim sırasında parametrelerini (ağırlıklarını) güncellemek için kullanılır.

'adam' (Adaptive Moment Estimation), yaygın olarak kullanılan ve genellikle başlangıç ​​olarak iyi performans gösteren bir optimizasyon algoritmasıdır. 
Ancak, başka optimizasyon algoritmaları da mevcuttur ve probleminize, veri kümenize veya ağınızın mimarisine bağlı olarak farklı optimizasyon 
algoritmalarının performansı değişebilir. İşte bazı yaygın optimizasyon algoritmaları ve kısa açıklamaları:

Stochastic Gradient Descent (SGD):

İteratif olarak veri kümesinin alt kümesini kullanarak gradyan inişini uygular.
optimizer='sgd' şeklinde kullanılır.
RMSprop:
Adapte edilmiş öğrenme oranına sahip bir gradyan iniş algoritmasıdır.
optimizer='rmsprop' şeklinde kullanılır.

Adagrad:
Her bir parametrenin öğrenme hızını bağımsız olarak ayarlar.
optimizer='adagrad' şeklinde kullanılır.

Adamax:
Adam'a dayanan bir algoritmadır, ancak normun maksimum yerine ortalama ile yeniden ölçeklendirilmesini kullanır.
optimizer='adamax' şeklinde kullanılır.

Nadam:
Nesterov hızlandırılmış gradyanı Adam'a dayalı bir algoritmadır.
optimizer='nadam' şeklinde kullanılır.

Bu optimizasyon algoritmaları arasında seçim yaparken genellikle deneme yanılma yöntemine başvurulur. Farklı optimizasyon algoritmaları, 
ağınızın konverjans hızı, başlangıç performansı, kaynak kullanımı ve diğer faktörler açısından farklı sonuçlar verebilir. 
Başlangıçta genellikle 'adam' kullanmak iyi bir seçenektir, ancak modelinizin performansını iyileştirmek için diğer algoritmaları da deneyebilirsiniz.
"""

print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))