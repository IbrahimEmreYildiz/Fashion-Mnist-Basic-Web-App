# Fashion MNIST Classifier (PyTorch & Streamlit)

[🇹🇷 Türkçe versiyon için aşağıya kaydırın / Scroll down for Turkish version](#fashion-mnist-sınıflandırıcı-pytorch--streamlit-türkçe)

This project contains a Convolutional Neural Network (CNN) model trained using PyTorch and an interactive Streamlit web application that visualizes the predictions of this model. The model is trained on the Fashion MNIST dataset and can successfully classify 10 different clothing categories.

## 📂 Project Structure

```text
fashion_mnist/
├── data/
│   └── fashion-mnist_test.csv     # Test dataset (Images and labels)
├── model/
│   └── fashion_mnist_model.pth    # Trained PyTorch model weights
├── src/
│   ├── app.py                     # Streamlit web application
│   ├── fashion_mnist.ipynb        # Jupyter Notebook containing model architecture and training
│   └── train_function.ipynb       # Jupyter Notebook containing training functions
├── .gitignore                     # Files to be ignored by Git
└── README.md                      # Project documentation (This file)
```

## 🛠️ Installation

Follow the steps below to run the project on your local machine:

1. **Install Requirements:**
   You need libraries such as PyTorch, Streamlit, and Pandas for the project to run.
   ```bash
   pip install torch torchvision pandas numpy streamlit
   ```

2. **Start the Application:**
   You can start the Streamlit application by running the following command in the terminal from the project root directory:
   ```bash
   streamlit run src/app.py
   ```

## 🚀 Usage

When the application opens:
1. Click the **"🎲 Rastgele Resim Seç ve Tahmin Et"** (Choose a Random Image and Predict) button.
2. The application will select a random image from the test dataset in the `data/` folder.
3. The selected 28x28 clothing image will be displayed on the screen.
4. The PyTorch model analyzes the image and makes a prediction.
5. On the result screen, the model's prediction is compared with the actual class. A green success message is displayed if correct, and a red error message if incorrect.

## 🧠 Class Names
The model can predict the following 10 classes:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

---

# Fashion MNIST Sınıflandırıcı (PyTorch & Streamlit) [Türkçe]

Bu proje, PyTorch kullanılarak eğitilmiş bir Convolutional Neural Network (CNN) modelini ve bu modelin tahminlerini görselleştiren etkileşimli bir Streamlit web uygulamasını içermektedir. Model, Fashion MNIST veri seti üzerinde eğitilmiştir ve 10 farklı giysi kategorisini başarıyla sınıflandırabilmektedir.

## 📂 Proje Yapısı

```text
fashion_mnist/
├── data/
│   └── fashion-mnist_test.csv     # Test veri seti (Görüntüler ve etiketler)
├── model/
│   └── fashion_mnist_model.pth    # Eğitilmiş PyTorch model ağırlıkları
├── src/
│   ├── app.py                     # Streamlit web uygulaması
│   ├── fashion_mnist.ipynb        # Model mimarisinin ve eğitimin yer aldığı Jupyter Notebook
│   └── train_function.ipynb       # Eğitim fonksiyonlarının yer aldığı Jupyter Notebook
├── .gitignore                     # Git tarafından yok sayılacak dosyalar
└── README.md                      # Proje dokümantasyonu (Bu dosya)
```

## 🛠️ Kurulum

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. **Gereksinimleri Kurun:**
   Projenin çalışması için PyTorch, Streamlit ve Pandas gibi kütüphanelere ihtiyacınız vardır.
   ```bash
   pip install torch torchvision pandas numpy streamlit
   ```

2. **Uygulamayı Başlatın:**
   Terminal üzerinden proje ana dizinindeyken aşağıdaki komutu çalıştırarak Streamlit uygulamasını başlatabilirsiniz:
   ```bash
   streamlit run src/app.py
   ```

## 🚀 Kullanım

Uygulama açıldığında:
1. **"🎲 Rastgele Resim Seç ve Tahmin Et"** butonuna tıklayın.
2. Uygulama, `data/` klasöründeki test veri setinden rastgele bir resim seçecektir.
3. Seçilen 28x28 boyutundaki giysi resmi ekranda gösterilir.
4. PyTorch modeli görüntüyü analiz eder ve bir tahminde bulunur.
5. Sonuç ekranında modelin tahmini ile gerçek sınıf karşılaştırılır. Doğruysa yeşil bir başarı mesajı, yanlışsa kırmızı bir hata mesajı görüntülenir.

## 🧠 Sınıf İsimleri
Model aşağıdaki 10 sınıfı tahmin edebilmektedir:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot
