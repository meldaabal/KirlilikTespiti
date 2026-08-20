import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const WasteDetectionApp());
}

class WasteDetectionApp extends StatelessWidget {
  const WasteDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Atık Tespit Uygulaması',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? selectedImage;
  final ImagePicker picker = ImagePicker();

  bool isLoading = false;
  String? predictedClass;
  double? confidence;

  Interpreter? interpreter;
  List<String> labels = [];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset(
      'assets/mobilenetv2_transfer_learning.tflite',
    );

    final labelsData = await rootBundle.loadString('assets/labels.txt');
    labels = labelsData
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();

    debugPrint("TFLite model yüklendi.");
    debugPrint("Labels: $labels");
  }

  Future<void> pickImageFromGallery() async {
    final XFile? image = await picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 85,
    );

    if (image != null) {
      setState(() {
        selectedImage = File(image.path);
        predictedClass = null;
        confidence = null;
      });
    }
  }

  Future<void> pickImageFromCamera() async {
    final XFile? image = await picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 85,
    );

    if (image != null) {
      setState(() {
        selectedImage = File(image.path);
        predictedClass = null;
        confidence = null;
      });
    }
  }

  Future<void> analyzeImage() async {
    if (selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Lütfen önce bir fotoğraf seçin.")),
      );
      return;
    }

    if (interpreter == null || labels.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Model henüz yüklenmedi.")),
      );
      return;
    }

    setState(() {
      isLoading = true;
      predictedClass = null;
      confidence = null;
    });

    try {
      final imageBytes = await selectedImage!.readAsBytes();
      final originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception("Görüntü okunamadı.");
      }

      final resizedImage = img.copyResize(
        originalImage,
        width: 224,
        height: 224,
      );

      final input = List.generate(
        1,
            (_) => List.generate(
          224,
              (y) => List.generate(
            224,
                (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r / 255.0,
                pixel.g / 255.0,
                pixel.b / 255.0,
              ];
            },
          ),
        ),
      );

      final output = List.generate(1, (_) => List.filled(6, 0.0));
      interpreter!.run(input, output);

      final scores = output[0];

      int maxIndex = 0;
      double maxScore = scores[0];

      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > maxScore) {
          maxScore = scores[i];
          maxIndex = i;
        }
      }

      setState(() {
        predictedClass = labels[maxIndex];
        confidence = maxScore * 100;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("TFLite analiz hatası: $e")),
      );
    }

    setState(() {
      isLoading = false;
    });
  }

  String getTurkishClassName(String? className) {
    switch (className) {
      case "cardboard":
        return "Karton";
      case "glass":
        return "Cam";
      case "metal":
        return "Metal";
      case "paper":
        return "Kağıt";
      case "plastic":
        return "Plastik";
      case "trash":
        return "Genel Çöp";
      default:
        return "-";
    }
  }

  String getRecommendation(String? className) {
    switch (className) {
      case "cardboard":
        return "Karton atıklar geri dönüşüm kutusuna atılmalıdır.";
      case "glass":
        return "Cam atıklar cam geri dönüşüm kutusuna atılmalıdır.";
      case "metal":
        return "Metal atıklar geri dönüşüme kazandırılmalıdır.";
      case "paper":
        return "Kağıt atıklar kağıt geri dönüşüm kutusuna atılmalıdır.";
      case "plastic":
        return "Plastik atıklar plastik geri dönüşüm kutusuna atılmalıdır.";
      case "trash":
        return "Bu atık genel çöp olarak değerlendirilmiştir.";
      default:
        return "";
    }
  }

  String getEnvironmentalImpact(String? className) {
    switch (className) {
      case "cardboard":
        return "Karton geri dönüşümü doğal kaynak tüketimini azaltır ve atık miktarını düşürür.";
      case "glass":
        return "Cam atıklar doğru şekilde ayrıştırıldığında tekrar tekrar geri dönüştürülebilir.";
      case "metal":
        return "Metal atıklar doğada uzun süre kalabilir ancak geri dönüştürülerek yeniden kullanılabilir.";
      case "paper":
        return "Kağıt geri dönüşümü ağaç kesimini azaltır ve doğal kaynakların korunmasına katkı sağlar.";
      case "plastic":
        return "Plastik atıklar doğada yüzlerce yıl kalabilir, bu yüzden geri dönüşüme kazandırılması önemlidir.";
      case "trash":
        return "Karışık atıklar geri dönüşüm verimini düşürür, bu nedenle atıkların ayrıştırılması gerekir.";
      default:
        return "";
    }
  }

  IconData getWasteIcon(String? className) {
    switch (className) {
      case "cardboard":
        return Icons.inventory_2;
      case "glass":
        return Icons.wine_bar;
      case "metal":
        return Icons.settings;
      case "paper":
        return Icons.description;
      case "plastic":
        return Icons.local_drink;
      case "trash":
        return Icons.delete;
      default:
        return Icons.recycling;
    }
  }

  Color getWasteColor(String? className) {
    switch (className) {
      case "cardboard":
        return Colors.brown;
      case "glass":
        return Colors.teal;
      case "metal":
        return Colors.blueGrey;
      case "paper":
        return Colors.green;
      case "plastic":
        return Colors.blue;
      case "trash":
        return Colors.redAccent;
      default:
        return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xfff2f7f2),
      appBar: AppBar(
        title: const Text("Atık Tespit Uygulaması"),
        centerTitle: true,
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 12),
            const Icon(Icons.recycling, size: 75, color: Colors.green),
            const SizedBox(height: 8),
            const Text(
              "Karada Çevre Kirliliği Tespiti",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 23, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              "Fotoğraf yükleyerek atık türünü yapay zekâ modeli ile analiz edin.",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 15, color: Colors.black54),
            ),
            const SizedBox(height: 22),
            Container(
              height: 250,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: Colors.green.shade200),
              ),
              child: selectedImage == null
                  ? const Center(
                child: Text(
                  "Henüz fotoğraf seçilmedi",
                  style: TextStyle(fontSize: 16, color: Colors.black54),
                ),
              )
                  : ClipRRect(
                borderRadius: BorderRadius.circular(18),
                child: Image.file(selectedImage!, fit: BoxFit.cover),
              ),
            ),
            const SizedBox(height: 18),
            ElevatedButton.icon(
              onPressed: pickImageFromGallery,
              icon: const Icon(Icons.photo_library),
              label: const Text("Galeriden Fotoğraf Seç"),
            ),
            const SizedBox(height: 10),
            ElevatedButton.icon(
              onPressed: pickImageFromCamera,
              icon: const Icon(Icons.camera_alt),
              label: const Text("Kameradan Fotoğraf Çek"),
            ),
            const SizedBox(height: 10),
            ElevatedButton.icon(
              onPressed: isLoading ? null : analyzeImage,
              icon: const Icon(Icons.search),
              label: Text(isLoading ? "Analiz Ediliyor..." : "Analiz Et"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 15),
              ),
            ),
            const SizedBox(height: 20),
            if (isLoading) const Center(child: CircularProgressIndicator()),
            if (predictedClass != null)
              Card(
                elevation: 6,
                color: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          CircleAvatar(
                            radius: 26,
                            backgroundColor:
                            getWasteColor(predictedClass).withOpacity(0.15),
                            child: Icon(
                              getWasteIcon(predictedClass),
                              color: getWasteColor(predictedClass),
                              size: 30,
                            ),
                          ),
                          const SizedBox(width: 14),
                          Expanded(
                            child: Text(
                              "${getTurkishClassName(predictedClass)} Atık Tespit Edildi",
                              style: TextStyle(
                                fontSize: 21,
                                fontWeight: FontWeight.bold,
                                color: getWasteColor(predictedClass),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: getWasteColor(predictedClass).withOpacity(0.08),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.verified, color: Colors.green),
                            const SizedBox(width: 10),
                            Text(
                              "Güven Oranı: %${confidence?.toStringAsFixed(2)}",
                              style: const TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                      if (confidence != null && confidence! < 70)
                        Container(
                          margin: const EdgeInsets.only(top: 12),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.orange.withOpacity(0.12),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.orange),
                          ),
                          child: const Row(
                            children: [
                              Icon(Icons.warning, color: Colors.orange),
                              SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  "Model bu görüntüden tam emin değil. Daha net ve yakın bir fotoğraf çekiniz.",
                                ),
                              ),
                            ],
                          ),
                        ),
                      const SizedBox(height: 16),
                      const Text(
                        "Geri Dönüşüm Önerisi",
                        style: TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        getRecommendation(predictedClass),
                        style: const TextStyle(
                          fontSize: 15,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 14),
                      const Text(
                        "Çevresel Etki",
                        style: TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        getEnvironmentalImpact(predictedClass),
                        style: const TextStyle(
                          fontSize: 15,
                          color: Colors.black54,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}