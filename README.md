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
- **Adding regularization layers**
- **K-fold cross-validation**
- **Grid sreach**

After re-reading the code, there seems to be some Data Leakage...
To prevent this, we will try to :
- **Hash before resizing to detect true duplicates**

## Technology Stack

### Deep Learning Framework
- **TensorFlow 2.x** - Core deep learning framework
- **Keras** - High-level API for model construction
- **ResNet50** - Pre-trained convolutional neural network architecture

### Experiment Tracking & Model Registry
- **MLflow** - Tracking hyperparameters, metrics, and model versions
- **MLflow Model Registry** - Versioning and staging trained models

### Data Processing
- **OpenCV (cv2)** - Image loading and preprocessing
- **NumPy** - Array operations and data manipulation
- **Hashlib (MD5)** - Duplicate image detection and removal

### Visualization
- **Matplotlib** - Training curves and result visualization
- **Seaborn** - Confusion matrix visualization

### Model Training Utilities
- **Scikit-learn** - Train/test splitting, stratification, metrics calculation
- **ImageDataGenerator** - Real-time data augmentation
- **EarlyStopping** - Prevent overfitting by monitoring validation loss
- **ReduceLROnPlateau** - Adaptive learning rate scheduling

### Hyperparameter Optimization
- **GridSearchCV** - Systematic hyperparameter tuning
- **K-fold Cross-Validation** - Robust model evaluation

### Hardware Acceleration
- **Multi-GPU support** - TensorFlow MirroredStrategy for distributed training
- **CUDA/cuDNN** - GPU acceleration for faster training

### Model Serialization
- **HDF5 format (.h5)** - Model saving and loading
- **Timestamp-based versioning** - Automatic model checkpointing

## License

Released under the **MIT License** to promote open collaboration while maintaining author credit.
