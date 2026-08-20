import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ana veri seti yolu
DATASET_PATH = "../dataset/raw/Garbage classification"

# Sonuçların kaydedileceği klasör
RESULTS_PATH = "../results"

# Sınıflar
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Görsel uzantıları
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def create_image_dataframe():
    image_paths = []
    labels = []

    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)

        for image_name in os.listdir(class_path):
            if image_name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(class_path, image_name))
                labels.append(class_name)

    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })

    return df


def save_distribution_plot(df, title, save_path):
    counts = df["label"].value_counts().reindex(CLASSES)

    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.xlabel("Sınıf")
    plt.ylabel("Görüntü Sayısı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    print("Veri hazırlama işlemi başladı...\n")

    df = create_image_dataframe()

    print("Toplam görüntü sayısı:", len(df))
    print("\nGenel sınıf dağılımı:")
    print(df["label"].value_counts())

    # Önce train+val ve test ayırıyoruz
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["label"]
    )

    # Sonra train ve validation ayırıyoruz
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15,
        random_state=42,
        stratify=train_val_df["label"]
    )

    print("\nTrain görüntü sayısı:", len(train_df))
    print("Validation görüntü sayısı:", len(val_df))
    print("Test görüntü sayısı:", len(test_df))

    print("\nTrain dağılımı:")
    print(train_df["label"].value_counts())

    print("\nValidation dağılımı:")
    print(val_df["label"].value_counts())

    print("\nTest dağılımı:")
    print(test_df["label"].value_counts())

    # CSV olarak kaydet
    train_df.to_csv(os.path.join(RESULTS_PATH, "tables", "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(RESULTS_PATH, "tables", "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(RESULTS_PATH, "tables", "test_data.csv"), index=False)

    # Grafik olarak kaydet
    save_distribution_plot(
        train_df,
        "Train Veri Seti Sınıf Dağılımı",
        os.path.join(RESULTS_PATH, "graphs", "train_distribution.png")
    )

    save_distribution_plot(
        val_df,
        "Validation Veri Seti Sınıf Dağılımı",
        os.path.join(RESULTS_PATH, "graphs", "val_distribution.png")
    )

    save_distribution_plot(
        test_df,
        "Test Veri Seti Sınıf Dağılımı",
        os.path.join(RESULTS_PATH, "graphs", "test_distribution.png")
    )

    print("\nCSV dosyaları kaydedildi:")
    print("results/tables/train_data.csv")
    print("results/tables/val_data.csv")
    print("results/tables/test_data.csv")

    print("\nGrafikler kaydedildi:")
    print("results/graphs/train_distribution.png")
    print("results/graphs/val_distribution.png")
    print("results/graphs/test_distribution.png")

    print("\nVeri hazırlama işlemi tamamlandı.")


if __name__ == "__main__":
    main()