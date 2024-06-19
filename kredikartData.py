"""
Kredi kartı kümeleme, kredi kartı sahiplerini satın alma alışkanlıklarına, kredi limitlerine ve diğer birçok finansal faktöre göre gruplandırma görevidir.
"""
import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt

data = pd.read_csv("../data/KrediKartData.csv")
# print(data.head())
# print(data.isnull().sum())

data = data.dropna() # Boş verileri sildim.
# print(data.isnull().sum())
"""
kredi kartı segmentasyonu görevi için veri setinde çok değerli olan üç özellik şunlardır:

BAKİYE: Kredi kartı müşterilerinin hesaplarında kalan bakiyedir.
SATIN ALIMLAR: Kredi kartı müşterilerinin hesaplarından yapılan alışveriş tutarıdır.
CREDIT_LIMIT: Kredi kartı limiti.

Bu üç özellik, kredi kartı sahiplerinin satın alma geçmişini, banka bakiyesini ve kredi limitini bize anlattıkları 
için kredi kartı sahiplerini gruplandırmaya yeterlidir. Şimdi veri kümesinden kümeler oluşturmak için bu özellikleri kullanalım:
"""

clustering_data = data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
from sklearn.preprocessing import MinMaxScaler
for i in clustering_data.columns:
    MinMaxScaler(i)
    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(clustering_data)
data["CREDIT_CARD_SEGMENTS"] = clusters

"""
“CREDIT_CARD_SEGMENTS” olarak yeni bir sütun ekledim. Kredi kartı müşterisi grubuna ilişkin etiketleri içerir. 
Oluşturulan gruplar 0 ile 4 arasında değişmektedir. Kolaylık olması açısından bu kümelerin adlarını dönüştüreceğim
"""

data["CREDIT_CARD_SEGMENTS"] = data["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1: 
    "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"})
print(data["CREDIT_CARD_SEGMENTS"].head(10))

import plotly.graph_objects as go
PLOT = go.Figure()
for i in list(data["CREDIT_CARD_SEGMENTS"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = data[data["CREDIT_CARD_SEGMENTS"]== i]['BALANCE'],
                                y = data[data["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z = data[data["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'BALANCE', titlefont_color = 'black'),
                                yaxis=dict(title = 'PURCHASES', titlefont_color = 'black'),
                                zaxis=dict(title = 'CREDIT_LIMIT', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12)).show()

