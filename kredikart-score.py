import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "plotly_white"

data = pd.read_csv("../data/credit-card-score.csv")
# print(data.head())
# print(data.info())
# print(data.isnull().sum())  # Kategorik boş sayısı
# print(data.isnull().sum().sum()) # Toplam boş sayısı

credits = data["Credit_Score"].value_counts()
# print(credits)

# fig = px.box(data,  x = "Occupation", 
#                     color = "Credit_Score",
#                     title = "Credit Scores Based on Occupation",
#                     color_discrete_map = {'Poor':'red',
#                                           'Standart':'yellow',
#                                           'Good':'green'})
# fig.show()

# fig = px.box(data, 
#              x="Credit_Score", ## Sütun adı
#              y="Annual_Income", ## Sütun adı
#              color="Credit_Score", # Sütun adı
#              title="Credit Scores Based on Annual Income", 
#              color_discrete_map={'Poor':'red',
#                                  'Standard':'yellow',
#                                  'Good':'green'})
# fig.update_traces(quartilemethod="exclusive")
# fig.show()

fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Inhand_Salary", 
             color="Credit_Score",
             title="Credit Scores Based on Monthly Inhand Salary", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()