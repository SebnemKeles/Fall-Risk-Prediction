# 🏥 Fall Risk Prediction Using Machine Learning and Deep Learning


## 📄 Abstract

This study investigates the application of data science in healthcare, focusing on predicting patient fall risk using a combination of machine learning (ML) and deep learning (DL) techniques. Gait videos were analyzed with OpenPose, a powerful computer vision tool for human pose estimation, to extract features that describe posture and movement patterns.

Multiple ML models, including k-Nearest Neighbor (kNN) and Support Vector Machine (SVM), were benchmarked against deep learning models such as Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN). Although DL models outperformed ML models, ensemble techniques, particularly Gradient Boosting, achieved the best performance with a **92% accuracy** and F1-scores of **0.91**, highlighting their potential for robust clinical applications.


## 📂 Dataset Description

The study utilized a public dataset from the Mendeley Data website, consisting of 188 videos categorized into three subfolders: Knee Osteoarthritis (KOA), Parkinson’s Disease (PD) and Normal/Healthy (NM) individuals with early (EL), mild (ML), moderate (MD), and severe (SV) categories. The dataset comprises 94 participants, including 60 women and 34 men, with an average age of approximately 60 years and an average height of 1.60 meters.

![image](https://github.com/user-attachments/assets/10297bd1-1a1c-4a86-b1ee-f20a64017432)


## 🔧 Tools & Libraries

- Jupyter Notebook
- Python
- NumPy / Pandas / Matplotlib / Seaborn
- OpenCV 
- scikit-learn
- TensorFlow / Keras


## 🚀 Workflow

1. **Preprocessing**
- Pose estimation using OpenPose  
- Reshaping and Standardization

2.  **Model Training**
- ML Models: k-Nearest Neighbor (kNN), Support Vector Machine (SVM)  
- DL Models: Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN)  
- Ensemble Model: Random Forest, Gradient Boosting

3. **Evaluation**
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix  
- Cross-validation  
- Hyperparameter tuning (GridSearchCV, manual tuning for DL)

  
## 📈 Results

| Model                              | Accuracy |
|------------------------------------|----------|
| K Nearest Neighbor (kNN)           | 72%      |
| Support Vector Machine (SVM)       | 64%      |
| Convolutional Neural Network (CNN) | 84%      |
| Long Short Term Memory (LSTM)      | 86%      |
| Random Forest Classifier           | 91%      |
| Gradient Boosting  Classifier      | **92%**  |

- Ensemble methods significantly improved performance  
- Precision, Recall, and F1-Score: ~0.91 for best model  


## 📷 Visual Samples

- Confusion Matrices  
- Feature visualizations from OpenPose  
- Classification reports


## 💡 Future Improvements

- Incorporate larger and more diverse datasets  
- Real-time video input support  
- Exploring alternative algorithms


✅ Status

✔️ Completed (MSc Dissertation Project).
