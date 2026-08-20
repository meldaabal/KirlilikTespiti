import os
import matplotlib.pyplot as plt

# Veri seti yolu
dataset_path = r"../dataset/raw/Garbage classification/"

print("Veri seti kontrol ediliyor...\n")

# Klasör var mı?
if not os.path.exists(dataset_path):
    print("HATA: Veri seti yolu bulunamadı!")
    exit()

# Sınıf klasörleri
classes = os.listdir(dataset_path)

print("Bulunan sınıflar:")
print(classes)
print()

image_counts = {}

total_images = 0

# Her klasörü gez
for class_name in classes:
    class_folder = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_folder):
        count = len(os.listdir(class_folder))
        image_counts[class_name] = count
        total_images += count
        print(f"{class_name}: {count} resim")

print("\nToplam resim sayısı:", total_images)

# Grafik çiz
plt.figure(figsize=(10,5))
plt.bar(image_counts.keys(), image_counts.values())
plt.title("Sınıflara Göre Görüntü Sayısı")
plt.xlabel("Sınıf")
plt.ylabel("Resim Sayısı")
plt.xticks(rotation=45)
plt.tight_layout()

# Kaydet
plt.savefig("../results/graphs/class_distribution.png")
print("\nGrafik kaydedildi:")
print("results/graphs/class_distribution.png")

plt.show()