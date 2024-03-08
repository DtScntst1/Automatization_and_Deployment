import streamlit as st
import joblib
import pandas as pd

# Uygulama başlığı
st.title("Ev Fiyatı Tahmini Uygulaması")

# Başlık
st.header("Ev Fiyatını Tahmin Edin")

# Alt Başlık
st.subheader("Bu uygulamayı kullanarak evinizin tahmini değerini öğrenin.")

# Görsel Ekleme
st.image("house.jpg", caption="Ev", width=400)

# Açıklama Ekleme
st.write("İlgilendiğiniz evin tahmini piyasa değerini aşağıda görebilirsiniz.")

# Modeldeki özellikleri yükleme
columns = ['GarageType', 'BsmtFinSF1', 'ExterQual', 'LandContour', 'FireplaceQu',
           'TotalBsmtSF', 'GrLivArea', 'KitchenQual', 'CentralAir_Y', 'GarageCars', 'OverallQual']

# Giriş Kutuları ve Slider'ları Tanımlama
garaj_tipi = st.selectbox("Garaj Tipi:", ["Bağlı", "Ayrık", "Yok"])
zemin_kat_bitmis_alan = st.slider("Zemin Kat Bitmiş Alanı (m²):", min_value=0, max_value=3000)
dis_kalite = st.selectbox("Dış Kalite:", ["Kötü", "Orta", "İyi"])
arazi_konturu = st.selectbox("Arazi Konturu:", ["Düşük", "Yüksek ve Dik", "Kıyı", "Düz"])
sobele_kalite = st.selectbox("Şömine Kalitesi:", ["Kötü", "Orta", "İyi"])
toplam_bodrum_kat_alani = st.slider("Toplam Bodrum Kat Alanı (m²):", min_value=0, max_value=3000)
yasam_alani = st.slider("Yaşam Alanı (m²):", min_value=500, max_value=6000)
mutfak_kalitesi = st.selectbox("Mutfak Kalitesi:", ["Kötü", "Orta", "İyi"])
merkezi_klima_var_mi = st.selectbox("Merkezi Klima Var mı?", ["Hayır", "Evet"])
garaj_kapasitesi = st.slider("Garaj Kapasitesi:", min_value=0, max_value=5)
genel_kalite_degeri = st.slider("Genel Kalite Değeri:", min_value=1, max_value=10)

# Örnek veri oluşturma
ornek_veri = {
    'GarageType': [garaj_tipi],
    'BsmtFinSF1': [zemin_kat_bitmis_alan],
    'ExterQual': [dis_kalite],
    'LandContour': [arazi_konturu],
    'FireplaceQu': [sobele_kalite],
    'TotalBsmtSF': [toplam_bodrum_kat_alani],
    'GrLivArea': [yasam_alani],
    'KitchenQual': [mutfak_kalitesi],
    'CentralAir_Y': [1 if merkezi_klima_var_mi == "Evet" else 0],
    'GarageCars': [garaj_kapasitesi],
    'OverallQual': [genel_kalite_degeri]
}
df = pd.DataFrame(ornek_veri)

# Model için gerekli sütunları ayarlama
df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

# Model yükleme
model = joblib.load(open("xgb_model.joblib", "rb"))

# Modelin kullandığı sütun isimlerini kontrol etme
print("Model sütunları:", model.get_booster().feature_names)

# Tahmin yapmadan önce gerekli sütunları seçme
df = df[model.get_booster().feature_names]

# Tahmin Butonu
if st.button('Tahmin Yap'):
    tahmin = round(model.predict(df)[0], 2)
    st.write('Ev fiyat tahmini:', tahmin, ' $')
else:
    st.write('Hoşçakalın!')





