"""
Zaman Serisi Tahmini, geleceğe yönelik kararlar almak için zaman serisi verilerinin analiz edilmesi ve modellenmesi anlamına gelir. 
Zaman Serisi Tahmininin bazı uygulamaları hava tahmini, satış tahmini, iş tahmini, hisse senedi fiyatı tahmini vb.'dir. ARIMA modeli,
Zaman Serisi Tahmini için kullanılan popüler bir istatistiksel tekniktir. 

ARIMA, Otoregresif Entegre Hareketli Ortalama anlamına gelir. Zaman Serisi Verilerini tahmin etmek için kullanılan bir algoritmadır. 
ARIMA modellerinin ARIMA(p, d, q) gibi üç parametresi vardır. Burada p, d ve q şu şekilde tanımlanır:

p, değerlere (etiket sütunu) eklenmesi veya çıkarılması gereken gecikmeli değerlerin sayısıdır. ARIMA'nın otoregresif kısmını yakalar.
d, sabit bir sinyal üretmek için verilerin kaç kez farklılaşması gerektiğini temsil eder. Durağan veri ise d değeri 0, mevsimsel veri ise d değeri 1 olmalıdır. 
d, ARIMA'nın bütünleşik kısmını yakalar.
q, değerlere (etiket sütunu) eklenen veya çıkarılan hata terimi için gecikmeli değerlerin sayısıdır. ARIMA'nın hareketli ortalama kısmını yakalar.

Hisse SENEDİ ANALİZİ
"""

import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('GOOG', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
# print(data.tail())

data = data[["Date", "Close"]]
# print(data.head())

import matplotlib.pyplot as plt
"""
plt.style.use('fivethirtyeight') matplotlib kütüphanesi ile grafiğinize belirli bir stili uygulamanızı sağlar. '
fivethirtyeight' stil dosyası, popüler bir veri haberciliği ve analitik web sitesi olan FiveThirtyEight'in grafiklerinde 
sıkça kullanılan bir stilidir. Bu stil, genellikle net ve basit bir görünüm sunar ve verilerin öne çıkmasına yardımcı olacak şekilde tasarlanmıştır.

Bu stilin bazı özellikleri şunlardır:

Kalın çizgi stilleri ve belirgin renk paleti.
Eksen etiketleri ve başlıklar için geniş yazı tipleri.
Arkaplanın temiz ve sade olması, görsel verilerin dikkat dağıtmadan sunulmasını sağlar.
# """
plt.style.use('fivethirtyeight')
# plt.figure(figsize=(15, 10))
# plt.plot(data["Date"], data["Close"])
# plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data["Close"], 
                            model='multiplicative', period=30)
"""
Bu kod, zaman serisi verilerinizdeki mevsimsel, trend ve döngüsel bileşenleri çıkarmak için kullanılan seasonal_decompose fonksiyonunu çağırıyor 
gibi görünüyor. Bu fonksiyon, statsmodels kütüphanesinin bir parçasıdır ve yaygın olarak zaman serisi analizinde kullanılır.

İşte seasonal_decompose fonksiyonunun kullandığınız parametrelerine bir göz atalım:

data["Close"]: Analiz edilmek istenen zaman serisi verisinin kendisi. Burada "Close" olasılıkla finansal bir varlık 
(örneğin, hisse senedi) için günlük kapanış fiyatlarını içeren bir sütun adıdır.

model='multiplicative': Zaman serisinin bileşenlerini modellemek için kullanılacak model tipini belirtir. 
'multiplicative' model, mevsimsel, trend ve döngüsel bileşenlerin zamanla değişen bir oranla değiştiği durumlar için uygundur. 
Alternatif olarak 'additive' model, bu bileşenlerin doğrudan toplamı olarak modellemek için kullanılır.

freq=30: Mevsimsel bileşenin periyodunu belirtir. 
Burada, 30 gün olarak belirtilmiş, yani eğer veri günlük ise bir aylık mevsimsel döngüler aranacak demektir.
freq değişmiş, period olmuş
"""
# fig = plt.figure()  
# fig = result.plot()  
# fig.set_size_inches(15, 10)
# plt.show()

pd.plotting.autocorrelation_plot(data["Close"])
# plt.show()

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Close"], lags = 100)
# plt.show()

p, d, q = 5, 1, 2
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data["Close"], order=(p,d,q))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

predictions = fitted.predict()
print(predictions)

import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

predictions = model.predict(len(data), len(data)+10)
print(predictions)
data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")

plt.show()

