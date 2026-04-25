import streamlit as st
import torch
from torch import nn
import pandas as pd
import numpy as np
import os

# Ayarlar ve Sınıf İsimleri
st.set_page_config(page_title="Fashion MNIST Sınıflandırıcı", page_icon="👗", layout="centered")

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Model Mimarisi (Jupyter notebook'taki ile birebir aynı)
class FashionCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1
                     ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=2
            ) 
        )

        self.layer_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1
                     ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=6272, # 128 * 7 * 7
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.layer_block_2(self.layer_block_1(x)))

# Model Yükleme Fonksiyonu
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "model", "fashion_mnist_model.pth")
    # Modelin input_shape=1, hidden_units=128, output_shape=10 olarak eğitildiği biliniyor.
    model = FashionCNN(input_shape=1, hidden_units=128, output_shape=10)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Model dosyası bulunamadı: {model_path}")
        return None

# Veri Yükleme Fonksiyonu
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "fashion-mnist_test.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, dtype=np.float32)
        return df
    else:
        st.error(f"Veri dosyası bulunamadı: {data_path}")
        return None

# Ana UI
st.title("👗 Fashion MNIST Görüntü Sınıflandırıcı")
st.markdown("Bu uygulama, PyTorch kullanılarak eğitilmiş bir CNN (Convolutional Neural Network) modelinin test verisi üzerindeki tahminlerini göstermektedir.")

st.divider()

# Model ve veriyi yükle
model = load_model()
df = load_data()

if model is not None and df is not None:
    # Rastgele veri seçimi butonu
    if st.button("🎲 Rastgele Resim Seç ve Tahmin Et", use_container_width=True):
        # Rastgele bir satır seç
        random_idx = np.random.randint(0, len(df))
        row = df.iloc[random_idx]
        
        true_label_idx = int(row["label"])
        true_label_name = CLASS_NAMES[true_label_idx]
        
        # Sadece pixel verilerini alıp 28x28 formatına dönüştür
        pixel_values = row.drop("label").values
        image_28x28 = pixel_values.reshape(28, 28)
        
        # Streamlit üzerinde görüntülemek için [0, 255] aralığındaki değerleri numpy uint8 matrisine çeviriyoruz
        # Aynı zamanda renkli çıkmaması için tek kanallı/grayscale olduğunu belirtiyoruz.
        image_display = image_28x28.astype(np.uint8)
        
        # Model tahmini için tensöre dönüştür (1, 1, 28, 28) ve normalize et (0-1 arası)
        image_tensor = torch.tensor(image_28x28, dtype=torch.float32) / 255.0
        image_tensor = image_tensor.view(1, 1, 28, 28)
        
        # Görselleştirme (2 kolon)
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            # Grayscale olarak görüntüle (width belirterek çok büyük çıkmasını engelliyoruz)
            st.image(image_display, caption="Seçilen Resim (28x28)", width=150)
            
        with col2:
            st.markdown("### 🧠 Model Tahmini")
            with st.spinner("Model tahminde bulunuyor..."):
                with torch.inference_mode():
                    predictions = model(image_tensor)
                    predicted_idx = torch.argmax(predictions, dim=1).item()
                    predicted_name = CLASS_NAMES[predicted_idx]
            
            # Doğruluk Kontrolü ve Renkli Çıktı
            if predicted_idx == true_label_idx:
                st.success(f"**DOĞRU:** {true_label_name}")
            else:
                st.error(f"**YANLIŞ!**\n\nTahmin: **{predicted_name}** \n\nGerçek: **{true_label_name}**")
