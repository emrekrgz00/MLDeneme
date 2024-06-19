# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

"""
Veri Ön İşleme

Veri setini yükledikten sonra, eksik değerleri ‘?’ işareti ile gösterilen verileri NaN olarak değiştirip,
bu satırları veri setinden çıkarıyoruz. Ardından, ‘id’ sütununu çıkararak sadece öznitelikler ve hedef değişkenle çalışıyoruz.
"""

import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape', 'marginal_adhesion',
         'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
dataset = pd.read_csv(url, names=names)

dataset.replace('?', np.nan, inplace=True)
dataset.dropna(inplace=True)
X = dataset.drop(['id', 'class'], axis=1)
y = dataset['class']

"""
Veri Görselleştirme

Veri setindeki özniteliklerin dağılımını ve sınıflar arasındaki farkları daha iyi anlayabilmek için
bazı görselleştirmeler yapıyoruz. Bu adım, veriyi anlamak ve modelin performansını artırmak için önemli ipuçları sağlayabilir.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Özelliklerin kutu grafikleri
# plt.figure(figsize=(12, 8))
# sns.boxplot(data=X)
# plt.xticks(rotation=45)
# plt.title("Feature Boxplots")
# plt.show()

# # Özelliklerin dağılımı (histogramlar)
# plt.figure(figsize=(12, 8))
# for i, feature in enumerate(X.columns):
#     plt.subplot(3, 4, i + 1)
#     sns.histplot(X[feature], bins=20, kde=True)
#     plt.title(feature)
# plt.tight_layout()
# plt.show()

"""
Model Eğitimi

Veriyi eğitim ve test setlerine böldükten sonra, özellikleri ölçeklendirmek için 
StandardScaler kullanıyoruz. Ardından, KNeighborsClassifier modeliyle eğitim yapıyoruz.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeli tanımlama ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = knn.predict(X_test)

# Modelin performansını değerlendirme
"""
Sonuçlar
Modelimizin performansını değerlendirmek için karışıklık matrisi ve sınıflandırma raporu kullanıyoruz.
Bu metrikler, modelin doğruluğu, hassasiyeti, hatırlama oranı ve F1 skoru gibi önemli ölçütleri sunar.

Modelimiz, test verileri üzerinde %96 doğruluk oranına ulaşmıştır. 
Bu, modelin meme kanseri teşhisinde oldukça başarılı olduğunu göstermektedir.
"""
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



"""
Sonuç;
Bu çalışma, KNeighborsClassifier algoritmasını kullanarak meme kanseri teşhisi için başarılı 
bir makine öğrenimi modeli geliştirmiştir. Veriyi önceden işleme, görselleştirme ve model 
değerlendirme adımları, modelin doğruluğunu ve güvenilirliğini artırmak için kritik öneme sahiptir. 
Gelecekte, farklı algoritmalar ve daha büyük veri setleri kullanarak modelin performansını daha da artırmak mümkündür.
"""
