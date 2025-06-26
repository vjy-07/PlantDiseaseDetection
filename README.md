# 🌿 Plant Disease Detection using CNN

A deep learning project to identify plant leaf diseases using Convolutional Neural Networks (CNN). Trained on the **New Plant Disease Dataset**, this model helps automate disease detection in crops.

---

## 📁 Dataset

- **Source:** [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- Contains over **87,000+ images** across **38 plant disease categories**
- Organized into `train`, `valid`, and `test` directories

---

## 🚀 Project Workflow

### 1. 📚 Importing Required Libraries
- Imported essential libraries such as:
  - TensorFlow & Keras
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn

---

### 2. 🧼 Data Preprocessing
- Loaded and structured the dataset into training, validation, and testing sets
- Applied image augmentation (rotation, zoom, flip) to improve generalization
- Normalized pixel values using rescaling techniques

---

### 3. 🏗️ Model Building
- Designed a **custom CNN model** using Keras Sequential API
- Included convolutional layers, pooling layers, dropout, and dense layers
- Used ReLU activation and softmax for multi-class classification

---

### 4. 🧪 Model Training
- Trained the model on the training set
- Used `categorical_crossentropy` loss and `Adam` optimizer
- Validated the model after each epoch on the validation set

---

### 5. 📈 Model Evaluation
- Evaluated the model on the test set using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix
  - Classification report

---

### 6. 📊 Accuracy Visualization
- Visualized model training progress using:
  - Training vs. validation accuracy curve

---

### 7. 💾 Model Saving
- Saved the trained model in `.keras` format
- Can be reused for predictions or integrated into an application

---

## ✅ Results

- Achieved approximately **95% accuracy** on the validation data
- Model performs well across multiple disease categories
- Lightweight CNN with strong generalization ability

---

## 🗂️ Project Structure
```
plant-disease-detection/
│
├── train/ # Training images
├── valid/ # Validation images
├── test/ # Test images
│
├── Train_plant_disease.ipynb # Notebook for training the model
├── Test_plant_disease.ipynb # Notebook for evaluating the model
├── main.py # Script for loading model and making predictions
```

