# lung_cancer_classificition_Resnet50
After 16 versions, this repository presents the final version of a deep learning-based classifier for lung cancer, distinguishing between three carcinoma subtypes.

## Dataset

- **Name:** LC25000 – Lung and Colon Cancer Histopathological Images  
- **Original Source:**  
  Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. *Lung and Colon Cancer Histopathological Image Dataset (LC25000)*. arXiv:1912.12142v1 [eess.IV], 2019  
- **Download:**  
  [Kaggle link](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)  
- **Composition:**  
  15,000 images (5000 per class)

## Preprocessing

- Removal of duplicate images using image hashing (MD5)  
- Class balance maintained  
- Images resized to (224x224)  
- Normalization to [0, 1] range  
- Augmentation: rotation, zoom, flips (for training only)  
- Stratified split into train/val/test sets with no data leakage  

## Model

- Pre-trained ResNet50 (ImageNet weights)  
- Final layer adjusted for 3-class classification  
- Categorical cross-entropy loss  
- Adam optimizer  
- Accuracy and F1-score used for evaluation  

## Results

- **Validation Accuracy:** 98.66%  
- **Test Accuracy:** 99%  
- **Good possibility of OVERFITTING.**

## What next ?

After testing on Render, we observe clear overfitting.
We will try to address this issue with :
- **K-fold cross-validation**
- **Grid sreach**

## 📄 License

Released under the **MIT License** to promote open collaboration while maintaining author credit.
