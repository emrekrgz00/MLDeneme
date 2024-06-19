import pandas as pd
import sqlite3

# Excel dosyalarını okuma
customers = pd.read_excel("../data/customers.xlsx")
purchases = pd.read_excel("../data/purchases.xlsx")

# SQLite ile veritabanı oluşturma
conn = sqlite3.connect(':memory:')
customers.to_sql('customers', conn, index=False, if_exists='replace')
purchases.to_sql('purchases', conn, index=False, if_exists='replace')

# SQL ile veri setlerini birleştirme
query = '''
SELECT c.customer_id, c.name, c.age, c.city, p.purchase_id, p.amount, p.date
FROM  customers c
JOIN purchases p ON c.customer_id = p.customer_id
'''

combined_data = pd.read_sql(query, conn)
conn.close()
print(combined_data)

## MakineÖğrenmesi

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Özellikler(Öz nitelik) ve hedef değişkeni ayırma
X = combined_data[["age", "city"]]
y = combined_data["amount"]

# Kategorik verileri işleme
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), ['city'])
    ],
    remainder='passthrough'
)
"""
Burada hata almıştım, HATA;
ValueError: Found unknown categories ['Los Angeles'] in column 0 during transform

Bu hata eğitim ve test setlerinin farklı kategorik değerler içermesinden kaynaklanır. Veri setim küçük olduğu için karşılaştım. Bu sorunu geçmek için,
OneHotEncoder'ın handle_unknown parametresini kullanmam gerekmektedir.
"""

print(preprocessor)
# Veri setini eğitim ve test setlerine ayırma

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline oluşturma

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modeli eğitme
model.fit(X_train, y_train)

# Tahminler yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squarred Error: {mse}')
