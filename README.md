# Water Quality and Potability Classification

A comprehensive machine learning project for predicting water potability using various classification algorithms and advanced feature engineering techniques.

## Project Overview

This project implements multiple machine learning models to classify water samples as potable (safe to drink) or non-potable based on various water quality parameters. The goal is to achieve high accuracy in predicting water safety to help ensure public health.

## Target Performance
- **Ideal Goal:** 85%
- **Best model:** SVM 61.28%
- **Current Best Performance:** 66.62%

## Project Structure

```
Water Quality and Potability/
├── Lab Activity Binary Classification.ipynb  # Main analysis notebook
├── water_potability.csv                     # Dataset
```

## Dataset Features

The dataset contains the following water quality parameters:

1. **pH**: Acidity/alkalinity level (6.5-8.5 is ideal)
2. **Hardness**: Mineral content in water
3. **Solids**: Total dissolved solids (TDS)
4. **Chloramines**: Disinfectant levels
5. **Sulfate**: Sulfate concentration
6. **Conductivity**: Electrical conductivity
7. **Organic Carbon**: Organic carbon content
8. **Trihalomethanes**: Chemical compound levels
9. **Turbidity**: Water clarity measure
10. **Potability**: Target variable (0 = Not Potable, 1 = Potable)

## Machine Learning Models Implemented

### Basic Models
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble tree-based model
- **Support Vector Machine (SVM)**: Kernel-based classifier
- **XGBoost**: Gradient boosting framework

### Advanced Techniques
- **Voting Classifier**: Combines multiple models
- **Stacking Classifier**: Meta-learning approach
- **Custom Ensemble**: Weighted prediction averaging
- **Neural Network (MLP)**: Deep learning approach

## Feature Engineering & Optimization

### Data Preprocessing
- **Missing Value Imputation**: Median strategy
- **Feature Scaling**: StandardScaler normalization
- **Class Imbalance Handling**: SMOTE, ADASYN, BorderlineSMOTE

### Advanced Feature Engineering
- **Polynomial Features**: Interaction terms (degree=2)
- **Non-linear Transformations**: Log, sqrt, square transforms
- **Power Transformations**: Yeo-Johnson normalization
- **Feature Binning**: Discretization for pattern discovery
- **Feature Selection**: SelectFromModel with XGBoost

### Hyperparameter Optimization
- **RandomizedSearchCV**: Efficient parameter search
- **Cross-Validation**: 3-5 fold validation
- **Threshold Optimization**: Custom decision boundaries

## Performance Acceleration

### GPU Acceleration (Optional)
The notebook includes support for GPU acceleration:
- **RAPIDS cuML**: GPU-accelerated machine learning
- **CuPy**: GPU array processing
- **XGBoost GPU**: tree_method='gpu_hist'

### Speed Optimizations
- Reduced cross-validation folds (3-fold CV)
- RandomizedSearchCV instead of GridSearchCV
- Parallel processing with n_jobs=-1
- Efficient ensemble voting strategies

## Results & Metrics

The project evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Classification breakdown

## Installation & Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### Optional GPU Acceleration
```bash
# For NVIDIA GPUs with CUDA 12.x
pip install cupy-cuda12x rapids-cudf rapids-cuml
```

### PyTorch (for neural networks)
```bash
pip install torch torchvision torchaudio
```

## Running the Project

1. **Clone/Download** the repository
2. **Install dependencies** using the commands above
3. **Open** `Lab Activity Binary Classification.ipynb` in Jupyter/VS Code
4. **Run cells sequentially** from top to bottom
5. **Monitor progress** - advanced models may take several minutes

## Key Insights

### Data Characteristics
- **Class Imbalance**: More non-potable samples than potable
- **Missing Values**: Present in multiple features
- **Feature Correlations**: Some features show moderate correlation

### Model Performance Patterns
- **XGBoost**: Generally performs best for this dataset
- **Ensemble Methods**: Often improve over individual models
- **Feature Engineering**: Non-linear transforms help performance
- **Threshold Tuning**: Can significantly improve results

## Advanced Experiments

The notebook includes several advanced techniques:

1. **Multiple Sampling Strategies**: SMOTE variants and hybrid approaches
2. **Kernel Approximation**: Nystroem method for SVM scaling
3. **Stacked Generalization**: Multi-level ensemble learning
4. **Custom Ensemble**: Probability averaging with threshold optimization
5. **Feature Space Exploration**: Polynomial and non-linear feature creation

## Usage Tips

### For Faster Execution
- Skip visualization cells if running batch processing
- Use smaller parameter grids for quick testing
- Enable GPU acceleration if available
- Run only essential models for initial exploration

### For Better Performance
- Experiment with different SMOTE variants
- Try custom threshold optimization
- Use ensemble methods for final predictions
- Consider feature selection for noisy datasets

## Contributing

Feel free to:
- Add new feature engineering techniques
- Implement additional algorithms
- Optimize hyperparameter search spaces
- Improve visualization and reporting

This project is for educational and research purposes.

---

*This project demonstrates comprehensive machine learning workflows including data preprocessing, feature engineering, model selection, ensemble methods, and performance optimization for binary classification tasks.*
