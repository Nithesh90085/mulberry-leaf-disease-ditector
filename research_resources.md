# Research Resources — Mulberry Leaf Disease Detection

## Key Papers to Cite in Your IEEE Submission

### 1. CNN + XAI (Explainability)
**Title:** Explainable deep learning model for automatic mulberry leaf disease classification  
**Journal:** Frontiers in Plant Science, 2023  
**DOI:** https://doi.org/10.3389/fpls.2023.1175515  
**Key contribution:** PDS-CNN model, Grad-CAM visualization, Bangladesh sericulture dataset (leaf rust + leaf spot)

---

### 2. CNN-ViT Hybrid (Highest Accuracy)
**Title:** Mulberry leaf disease detection by CNN-ViT with XAI integration  
**Journal:** PLOS ONE, 2024  
**URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325188  
**Key contribution:** 95.60% accuracy, 0.0017s inference, Vision Transformer + CNN fusion

---

### 3. YOLOv8 Real-Time Detection
**Title:** Detection of Mulberry Leaf Diseases in Natural Environments Based on Improved YOLOv8  
**Journal:** MDPI Forests, 2024  
**URL:** https://www.mdpi.com/1999-4907/15/7/1188  
**Key contribution:** YOLOv8-RFMD for real-time field detection, small lesion localization

---

### 4. Multi-Scale ResNet + SENet
**Title:** Recognition of mulberry leaf diseases based on multi-scale residual network fusion SENet  
**Journal:** PLOS ONE, 2024  
**URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298700  
**Key contribution:** 98.72% accuracy, SENet attention mechanism, data augmentation techniques

---

### 5. YOLO-based Detection (Foundational)
**Title:** Mulberry leaf disease detection using YOLO  
**Source:** ResearchGate, 2021  
**URL:** https://www.researchgate.net/publication/353143046_Mulberry_leaf_disease_detection_using_YOLO  
**Key contribution:** Grid-based CNN+YOLO approach, early work in this domain

---

### 6. SVM vs PNN Comparison
**Title:** Comparative Analysis on Mulberry Leaf Disease Detection Using SVM and PNN  
**Publisher:** Springer, 2022  
**URL:** https://link.springer.com/10.1007/978-981-19-1484-3_16  
**Key contribution:** Statistical thresholding segmentation + PNN classification

---

### 7. State-Space Model Approach
**Title:** A State-Space Model-Based Approach for Mulberry Leaf Disease Detection  
**Journal:** MDPI Plants, 2025  
**URL:** https://www.mdpi.com/2223-7747/14/13/2084  
**Key contribution:** Novel state-space model architecture for disease detection

---

## Datasets

| Dataset | Source | Classes | Size |
|---------|--------|---------|------|
| PlantVillage | https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset | 38 | 87,900 images |
| Mulberry Leaf (Bangladesh) | Collected in Frontiers 2023 paper | 3 (Healthy, Rust, Spot) | ~2,000 images |
| Custom field dataset | Collect from local sericulture farms | 5 | Recommended: 500+ per class |

---

## IEEE Xplore Search
Search for latest IEEE papers:  
https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=mulberry+leaf+disease+detection

---

## Suggested Paper Structure (IEEE Format)

1. Abstract
2. Introduction — importance of sericulture, disease impact
3. Related Work — cite papers above
4. Methodology — MobileNetV2 transfer learning, data augmentation
5. Dataset — collection, preprocessing, augmentation
6. Experiments — training setup, hyperparameters
7. Results — accuracy, precision, recall, F1, confusion matrix
8. Discussion — comparison with SOTA
9. Conclusion
10. References

## Image Preprocessing Pipeline (cite in paper)
- Resize to 224×224 (MobileNetV2 input)
- Normalize pixel values to [0, 1]
- Augmentation: rotation ±30°, flip, zoom, brightness adjustment
- Train/Val/Test split: 70/20/10
