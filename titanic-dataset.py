import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score

veri = pd.read_csv("../data/Titanic-Dataset.csv")

# print(veri.columns)  ## ColumnsName
# for i in veri.columns: ## ColumnsName Sıralı
#     print(i)
"""
Column İsimleri --- Türkçe Halleri  
'YolcuID', 'HayattaKalma', 'Sinif','Isim', 'Cinsiyet', 'Yas', 'KardesEsi', 
                'EbeveynCocuk', 'Bilet', 'Ucret', 'Kabin', 
                'BinişLimani'

'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                'Parch', 'Ticket', 'Fare', 'Cabin',
                'Embarked'
"""
# # Kolon adı değiştirme
# df = pd.DataFrame(veri)
# print("Eski sütun isimleri:", df.columns)
# #Sütun isimlerini değiştirme
# df.rename(columns={'YolcuID':'PassengerId'}, inplace=True)
# print("Yeni sütun isimleri:", df.columns)


# print(veri.head())   # İlk 5 veri

# null değer var mı?

# nulls = veri.isnull().sum()
# print(nulls)

#yaş kolonundaki eksik verileri ortalam ile doldurdum 
ort = veri["Age"] ## column tamamı
ort = veri["Age"].isnull().sum() ## Columnda toplam boş sayısı
ort = veri["Age"].mean() ## Columns ortalaması
veri["Age"] = veri["Age"].fillna(ort)  ## Boş yerlere columns ortalamasını yazmak 
# ort = veri["Age"].isnull().sum() ## Columnda toplam boş sayısı
# print(ort)

### İsimlerdeki Ünvanları Ayırma ve Yeni Bir Sütuna Eklemek
veri["Unvan"] = veri["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
#mr missi doctor captan vb ünvanlara sahip kişilerin isismlerinin başında bu bilgiler vardı
#bubilgileri split kullanarak aldım
# print(veri["Unvan"])
# print(veri)

# sınıf_Dağilimi = veri["Pclass"].value_counts()
# BinişLimani = veri["Embarked"].value_counts()
# cinsiyet_sayilari = veri['Sex'].value_counts()
# plt.figure(figsize=(18, 7))
# plt.subplot(1, 3, 1) 
# plt.pie(sınıf_Dağilimi, labels=sınıf_Dağilimi.index, autopct='%1.1f%%', startangle=140)
# plt.title('Yolcuların Kullandıkları Sınıf Dağılımı')
# plt.subplot(1, 3, 2)
# plt.pie(BinişLimani, labels=BinişLimani.index, autopct='%1.1f%%', startangle=140)
# plt.title('Yolcuların BinişLimani Dağılımı')
# plt.subplot(1, 3, 3)
# plt.pie(cinsiyet_sayilari, labels=cinsiyet_sayilari.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'pink'])
# plt.title('Cinsiyet Dağılımı')
# plt.show()

# unvan_Dagılımı = veri["Unvan"].value_counts()
# plt.figure(figsize=(15, 10))
# unvan_Dagılımı.plot(kind='bar', color='GREEN')
# plt.xlabel('Ünvan',color="blue")
# plt.ylabel('Adet')
# plt.title('Gemideki kişlerin ünvanı')
# plt.show()

# plt.figure(figsize=(20, 10))
# sbn.histplot(veri["Unvan"], 
#              bins=17, kde=True,color="Green",
#              edgecolor='black',label="ünvanı dağılımı")
# plt.ylabel('Adet')
# plt.title('Gemideki kişlerin ünvanı')
# plt.show()

# dizi = {1:"Hayatta",0:"ölü"}
# plt.figure(figsize=(18, 7))
# plt.subplot(1,2,1)
# plt.title("Hayatta kalanlar ve ölenlerin dağilimi")
# veri["kontrol"] = veri["Survived"].map(dizi)
# sbn.countplot(data=veri,x="kontrol")
# plt.xlabel("")
# plt.ylabel("Sayi")
# plt.subplot(1,2,2)
# plt.title('Hayatta Kalanlar ve Kalmayanlar: Cinsiyete Göre Dağılım')
# sbn.countplot(data=veri,x="Sex",hue="Survived")
# plt.legend(title='Hayatta Kalma', labels=['Kalmayanlar', 'Kalanlar'])
# plt.show()

# toplam_erkek_adedi = len(veri.loc[veri["Sex"] == "male"])
# hayatta_kalan_erkek_adedi = len(veri[(veri["Sex"] == "male") & (veri["Survived"] == 1)])
# erkek_hayatta_kalma_orani = (hayatta_kalan_erkek_adedi / toplam_erkek_adedi) * 100
# print("Erkeklerin hayatta kalma oranı: {:.2f}%".format(erkek_hayatta_kalma_orani))
# print(toplam_erkek_adedi)
# print(hayatta_kalan_erkek_adedi)
# print(erkek_hayatta_kalma_orani)
# toplam_kadın_adedi = len(veri.loc[veri["Sex"] == "female"])
# hayatta_kalan_kadın_adedi = len(veri[(veri["Sex"] == "female") & (veri["Survived"] == 1)])
# kadın_hayatta_kalma_orani = (hayatta_kalan_kadın_adedi / toplam_kadın_adedi) * 100
# print("Kadınların hayatta kalma oranı: {:.2f}%".format(kadın_hayatta_kalma_orani))

# plt.figure(figsize=(10, 6))
# sbn.countplot(data=veri, x='Pclass', hue='Survived')
# plt.title('Hayatta Kalanlar ve Kalmayanlar: Sınıflara Göre Dağılım')
# plt.xlabel('Sınıf')
# plt.ylabel('Kişi Sayısı')
# plt.legend(title='Hayatta Kalma', labels=['Kalmayanlar', 'Kalanlar'])
# plt.show()

# plt.figure(figsize=(14,5))
# sbn.histplot(veri.loc[veri["Survived"]==1]["Age"], 
#              bins=50, kde=True,color="blue",
#              edgecolor='black',label="Hayatta kalnların yaş dağılımı")
# sbn.histplot(veri["Age"], bins=50, kde=True,
#              color="red",edgecolor='black',
#              alpha=0.1,label="Yaş dagılımı")
# plt.legend()
# plt.savefig("../figures/YaşÖlümGrafiği.png", dpi=300, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(12,8))
# sbn.scatterplot(data=veri, x="Fare", y="Age", hue="Survived")
# plt.title("Titanic: Ücret ve Yaşa Göre Hayatta Kalma")
# plt.xlabel("Ücret")
# plt.ylabel("Yaş")
# plt.show()

label_encoder = LabelEncoder() #string ifadelerin sayısal veriye çevrilmesini sağlar
scaler = MinMaxScaler()
# veri["Cinsiyet1"] = label_encoder.fit_transform(veri["Sex"])
# veri["BinişLimani1"] = label_encoder.fit_transform(veri["Embarked"])

######             Veri Setindeki Belirli Özellikler Arasındaki Korelasyon Haritası
# plt.figure(figsize=(16,10))
# corr = veri[["Survived","Pclass","Cinsiyet1","Age","SibSp","Parch","Fare","BinişLimani1"]].corr()
# sbn.heatmap(corr, cmap = "YlGnBu", annot=True, fmt=".2f")
# plt.savefig("../figures/CorrMap.png", dpi=300, bbox_inches="tight")
# plt.show()

###### Hayatta Kalma ile Diğer Özellikler Arasındaki Korelasyonun
# corr_matrix = corr.corr()["Survived"].drop("Survived")
# plt.figure(figsize=(8, 6))
# corr_matrix.plot(kind="bar", color="skyblue")
# plt.title("Hayatta Kalma ile Diğer Özellikler Arasındaki Korelasyon")
# plt.xlabel("Özellik")
# plt.ylabel("Korelasyon Katsayısı")
# plt.xticks(rotation=45)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.show()

#gereksiz veriler silinmiştir
veri.drop(columns=["Name","Ticket","Cabin"],inplace=True)
# print(veri.head())

# Cinsiyet ve  Unvan kolonları  label_encoder yapılmıştır
veri["Sex"] = label_encoder.fit_transform(veri["Sex"])
liman = pd.get_dummies(veri["Embarked"]).astype(int)
veri["Unvan"]= label_encoder.fit_transform(veri["Unvan"])
veri = pd.concat([veri, liman], axis=1)
veri.drop(columns=["PassengerId","Embarked"],inplace=True)
#MinMaxScaler yapılarak veriler 0-1 arsasına getirilmiştir 
# Normalizasyon yapıyor. Her bir sayıyı o verilere ait maksimum'a bölüyor ve 0 - 1 arasında dönüşüyor.
veri[['Age', 'Fare',"Pclass","Unvan"]] = scaler.fit_transform(veri[['Age', 'Fare',"Pclass","Unvan"]])
# print(veri.head())
x_değerleri = veri.drop(columns=["Survived"])
x_değerleri = x_değerleri.values
y_değeri = veri["Survived"].values
# print(x_değerleri)
# print(y_değeri)
# Eğitilmeye Hazır Veri Setimiz
# print(veri.head())
x_train,x_test,y_train,y_test = train_test_split(x_değerleri,y_değeri,test_size=0.2,random_state=14)

####   Sınıflandırma Algoritmalarının Performansının ROC Eğrileriyle İncelenmesi

### RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, rf_pred)
# print("Test seti doğruluğu:", accuracy)
cm = confusion_matrix(y_test, rf_pred)
# print("Confusion Matrix : ")
# print(cm) #gridsearch yapılmadan önce 
# print("Sınıflandırma Raporu:")
# print(classification_report(y_test, rf_pred))

### SVC
svm_model = SVC(probability=True ,kernel='linear')
svm_model.fit(x_train, y_train)
svm_tahmin= svm_model.predict(x_test)
conf_matrix = confusion_matrix(y_test, svm_tahmin)
# print("Confusion Matrix : ")
# print(conf_matrix)
accuracy = accuracy_score(y_test,svm_tahmin)
# print("Test seti doğruluğu:", accuracy) 
# print("Sınıflandırma Raporu:")
# print(classification_report(y_test, svm_tahmin))

### DecisionTreeClassifier
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)
tahmin_dec = dec.predict(x_test)
accuracy = accuracy_score(y_test,tahmin_dec)
# print("Test seti doğruluğu: ", accuracy)
conf_matrix = confusion_matrix(y_test, tahmin_dec)
# print("Confusion Matrix : ")
# print(conf_matrix)
# print("Sınıflandırma Raporu:")
# print(classification_report(y_test, tahmin_dec))

### LogisticRegression
log = LogisticRegression(max_iter=1000)
log.fit(x_train, y_train)
log_test = log.predict(x_test)
conf_matrix = confusion_matrix(y_test, log_test)
accuracy = accuracy_score(y_test, log_test)
# print("Confusion Matrix:")
# print(conf_matrix)
# print("Test seti doğruluğu: ", accuracy)
# print("Sınıflandırma Raporu:")
# print(classification_report(y_test, log_test))

### XGBClassifier
xgb_model = XGBClassifier(random_state=21)
xgb_model.fit(x_train,y_train)
model_tahmini_xgb = xgb_model.predict(x_test)
conf_matrix = confusion_matrix(y_test, model_tahmini_xgb)
accuracy = accuracy_score(y_test, model_tahmini_xgb)
# print("Confusion Matrix:")
# print(conf_matrix)
# print("Test seti doğruluğu: ", accuracy)
# print("Sınıflandırma Raporu:")
# print(classification_report(y_test, model_tahmini_xgb))

### KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)  
knn.fit(x_train, y_train)
knn_y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, knn_y_pred)
# print("Test seti doğruluğu: ",accuracy)
# print("Sınıflandırma Raporu:")
# print(classification_report(y_test, knn_y_pred))
# print("Confusion Matrix: ")
# print(confusion_matrix(y_test, knn_y_pred))

from sklearn.model_selection import GridSearchCV
# Makine öğrenimi modelleri için en iyi hiperparametreleri seçmeye yardımcı olan bir yöntemdir.
#GridSearchCV, makine öğrenimi modellerinde kullanılan bir hiperparametre optimizasyon tekniğidir. 
#Hiperparametreler, modelin yapılandırılmasını ve performansını etkileyen parametrelerdir ve genellikle model eğitimi sırasında manuel olarak ayarlanmaları gereklidir.

parametreler = {
    'n_estimators': [10, 100,1000],
    # Oluşturulacak olan ağaç sayısı. Büyük bir sayı, modelin genellikle daha iyi performans göstermesini sağlayabilir, ancak eğitim süresini artırabilir.
    'max_depth': [15,20,40,80,90],
    # Bir karar ağacının maksimum derinliği. Daha derin ağaçlar daha fazla ayrıntıya izin verir, ancak aşırı öğrenme riskini artırabilir.
    'max_features' :[0.5,1,2,3],
    # Her bir bölünme noktasında göz önünde bulundurulacak özellik sayısı. Küçük bir sayı, modelin genelleşmesine yardımcı olabilir.
    'min_samples_split': [8, 10,12],
    # Bir düğümü bölmek için gereken minimum örnek sayısı. Küçük değerler modelin aşırı uyumasına yol açabilir.
    'min_samples_leaf': [ 5,10,15],
    # Bir yaprak düğümünde gereken minimum örnek sayısı. Küçük değerler aşırı uyum riskini artırabilir.
    'criterion' : ['gini', 'entropy'],
    # Bölünmelerin kalitesini ölçmek için kullanılan fonksiyon. 'gini' veya 'entropy' seçeneklerinden biri olabilir.
    'bootstrap': [True]
    # Eğitim verisinin rastgele örneklemler alınıp alınmayacağını belirleyen bir parametre. True ise bootstrap örnekleme yapılır.
}
grid_model = GridSearchCV(estimator=rf_model, param_grid=parametreler, cv=5, scoring='accuracy', verbose=2)
grid_model.fit(x_train, y_train)
print("En iyi parametreler:", grid_model.best_params_)
print("En iyi skor:", grid_model.best_score_)
en_iyi_model = grid_model.best_estimator_
y_pred = en_iyi_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test seti doğruluğu:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(14, 9))

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(x_test)[:, 1])
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(x_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_model.predict_proba(x_test)[:, 1])
roc_auc_svm = roc_auc_score(y_test, svm_model.predict_proba(x_test)[:, 1])
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')

fpr_dec, tpr_dec, _ = roc_curve(y_test, dec.predict_proba(x_test)[:, 1])
roc_auc_dec = roc_auc_score(y_test, dec.predict_proba(x_test)[:, 1])
plt.plot(fpr_dec, tpr_dec, label=f'Decision Tree (AUC = {roc_auc_dec:.2f})')

fpr_log, tpr_log, _ = roc_curve(y_test, log.predict_proba(x_test)[:, 1])
roc_auc_log = roc_auc_score(y_test, log.predict_proba(x_test)[:, 1])
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(x_test)[:, 1])
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(x_test)[:, 1])
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(x_test)[:, 1])
roc_auc_knn = roc_auc_score(y_test, knn.predict_proba(x_test)[:, 1])
plt.plot(fpr_knn, tpr_knn, label=f'K-Nearest Neighbors (AUC = {roc_auc_knn:.2f})')

fpr_grid, tpr_grid, _ = roc_curve(y_test, grid_model.predict_proba(x_test)[:, 1])
roc_auc_grid = roc_auc_score(y_test, grid_model.predict_proba(x_test)[:, 1])
plt.plot(fpr_grid, tpr_grid, label=f'Grid Search Random Forest (AUC = {roc_auc_grid:.2f})')

plt.title('Farklı Modeller İçin ROC Eğrileri')
plt.savefig("../figures/MLROCeğrisi.png", dpi=300, )
plt.legend()
plt.grid(True)
plt.show()