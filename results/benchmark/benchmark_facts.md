# Benchmark facts for the manuscript

## Protocol (methods: technical validation)

- Split: splits/split_seed42.csv — train 1051 (normal 424, OA 627), val 225 (normal 91, OA 134), test 224 (normal 90, OA 134); look-alike groups kept in one subset
- Models: ResNet50, VGG16, DenseNet121, MobileNetV2, Xception, InceptionResNetV2; weights: imagenet
- Input 224×224 RGB, each model's own Keras preprocess_input
- Augmentation (training only): horizontal flip, rotation ±40°, translation ±10%, zoom ±10%, contrast ±20%
- Head: global average pooling, dropout 0.3, 2-unit softmax; balanced class weights
- Warm-up: backbone frozen, Adam lr 0.001, 5 epochs
- Fine-tuning: top 30% of backbone layers unfrozen (BatchNorm frozen), Adam lr 1e-05, up to 30 epochs, early stopping on val loss (patience 5, best weights restored), ReduceLROnPlateau (factor 0.2, patience 2)
- Loss: sparse categorical cross-entropy; batch size 32; seeds: 42, 43, 44
- Software: Python 3.12.3, TensorFlow 2.21.0, scikit-learn 1.9.1; hardware: Linux-6.8.0-138-generic-x86_64-with-glibc2.39; devices /physical_device:CPU:0, /physical_device:GPU:0

## Results (test set)

- Accuracy range: 0.8854–0.9435; highest: VGG16 (0.9435 ± 0.0136)
- ResNet50: acc 0.9107 ± 0.0134, macro F1 0.9085 ± 0.0133, AUC 0.9820 ± 0.0026, sensitivity 0.8930 ± 0.0240, specificity 0.9370 ± 0.0128 (TN 85, FP 5, FN 12, TP 122)
- VGG16: acc 0.9435 ± 0.0136, macro F1 0.9407 ± 0.0145, AUC 0.9839 ± 0.0032, sensitivity 0.9677 ± 0.0086, specificity 0.9074 ± 0.0257 (TN 83, FP 7, FN 5, TP 129)
- DenseNet121: acc 0.9137 ± 0.0068, macro F1 0.9103 ± 0.0079, AUC 0.9629 ± 0.0026, sensitivity 0.9254 ± 0.0129, specificity 0.8963 ± 0.0357 (TN 82, FP 8, FN 11, TP 123)
- MobileNetV2: acc 0.8854 ± 0.0068, macro F1 0.8834 ± 0.0066, AUC 0.9686 ± 0.0013, sensitivity 0.8507 ± 0.0149, specificity 0.9370 ± 0.0064 (TN 84, FP 6, FN 20, TP 114)
- Xception: acc 0.9211 ± 0.0144, macro F1 0.9187 ± 0.0143, AUC 0.9699 ± 0.0048, sensitivity 0.9154 ± 0.0283, specificity 0.9296 ± 0.0064 (TN 84, FP 6, FN 14, TP 120)
- InceptionResNetV2: acc 0.9330 ± 0.0118, macro F1 0.9300 ± 0.0118, AUC 0.9759 ± 0.0018, sensitivity 0.9552 ± 0.0269, specificity 0.9000 ± 0.0111 (TN 82, FP 8, FN 9, TP 125)

Positive class = osteoarthritis. Precision/recall/F1 are macro-averaged.
Full numbers: table2_baseline.csv, metrics_per_run.csv.
