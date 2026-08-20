# KirlilikTespiti

Mobil cihazlarda derin öğrenme tabanlı çevre kirliliği ve atık tespiti gerçekleştiren bir görüntü sınıflandırma uygulamasıdır.

Proje kapsamında atık görüntülerini sınıflandırmak için bir derin öğrenme modeli geliştirilmiş, eğitilen model mobil cihazlarda çalışabilecek şekilde TensorFlow Lite (TFLite) formatına dönüştürülmüş ve Flutter ile geliştirilen mobil uygulamaya entegre edilmiştir.

🎯 Projenin Amacı

Bu projenin amacı, kullanıcıların mobil cihazlarından bir atık fotoğrafı seçerek atığın türünü otomatik olarak belirleyebilmesini sağlamaktır.

Sistem, seçilen görüntüyü eğitilmiş derin öğrenme modeli ile analiz ederek aşağıdaki sınıflardan birine ayırır:

📦 Cardboard — Karton
🍶 Glass — Cam
🔩 Metal — Metal
📄 Paper — Kağıt
♻️ Plastic — Plastik
🗑️ Trash — Diğer Atık
🧠 Yapay Zekâ Modeli

Görüntü sınıflandırma işlemi için MobileNetV2 tabanlı transfer learning yaklaşımı kullanılmıştır.

Model geliştirme sürecinde farklı CNN mimarileri ve eğitim yöntemleri üzerinde deneyler gerçekleştirilmiş, mobil uygulamada kullanılmak üzere uygun model seçilmiştir.

Eğitilen model daha sonra:

mobilenetv2_transfer_learning.tflite

formatına dönüştürülerek Flutter uygulamasına entegre edilmiştir.

📱 Mobil Uygulama

Mobil uygulama Flutter ve Dart kullanılarak geliştirilmiştir.

Uygulama temel olarak şu şekilde çalışır:

Kullanıcı cihazından bir görüntü seçer.
Görüntü uygulama tarafından işlenir.
TensorFlow Lite modeli görüntüyü analiz eder.
Model atık sınıfını tahmin eder.
Tahmin sonucu ve güven değeri kullanıcıya gösterilir.

Model doğrudan mobil uygulamanın içerisinde çalıştığı için sınıflandırma işlemi için harici bir sunucuya ihtiyaç duyulmaz.

🛠️ Kullanılan Teknolojiler
Yapay Zekâ ve Python
Python
TensorFlow / Keras
MobileNetV2
Transfer Learning
NumPy
Pillow
TensorFlow Lite
FastAPI
Mobil Uygulama
Flutter
Dart
TensorFlow Lite
Android Studio
📂 Proje Yapısı
KirlilikTespiti/
│
├── python/
│   ├── api/
│   ├── app/
│   ├── models/
│   ├── results/
│   ├── src/
│   └── convert_to_tflite.py
│
├── mobile/
│   ├── android/
│   ├── assets/
│   ├── ios/
│   ├── lib/
│   ├── linux/
│   ├── macos/
│   ├── test/
│   ├── web/
│   ├── windows/
│   ├── pubspec.yaml
│   └── pubspec.lock
│
├── .gitignore
└── README.md
python/

Modelin hazırlanması, eğitilmesi, değerlendirilmesi ve TensorFlow Lite formatına dönüştürülmesiyle ilgili Python kodlarını içerir.

mobile/

Eğitilmiş TensorFlow Lite modelini kullanan Flutter mobil uygulamasını içerir.

📊 Veri Seti

Model aşağıdaki altı atık kategorisini sınıflandıracak şekilde geliştirilmiştir:

cardboard, glass, metal, paper, plastic, trash

Ham görüntü veri seti repository boyutunu gereksiz yere artırmamak amacıyla GitHub repository'sine dahil edilmemiştir.

🚀 Flutter Uygulamasını Çalıştırma

Projeyi klonladıktan sonra mobil uygulama klasörüne geçin:

cd mobile

Flutter bağımlılıklarını yükleyin:

flutter pub get

Bağlı bir Android cihaz veya emulator üzerinde uygulamayı çalıştırın:

flutter run
🔬 Model Geliştirme

Model geliştirme ve deney kodları:

python/src/

klasöründe bulunmaktadır.

Eğitim sonuçları ve değerlendirme çıktıları ise:

python/results/

klasöründe tutulmaktadır.

Mobil uygulamada kullanılan model:

mobile/assets/mobilenetv2_transfer_learning.tflite
🔄 Genel Sistem Akışı
Atık Görüntüsü
      ↓
Görüntü Ön İşleme
      ↓
MobileNetV2 Modeli
      ↓
TensorFlow Lite
      ↓
Flutter Mobil Uygulaması
      ↓
Atık Sınıfı + Güven Değeri
👩‍💻 Geliştirici

Bu proje, görüntü işleme ve derin öğrenme yöntemlerinin mobil uygulamalar üzerinde kullanılmasını incelemek amacıyla geliştirilmiştir.
