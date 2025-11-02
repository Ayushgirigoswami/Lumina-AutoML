# 🌟 Lumina AutoML

<div align="center">

![Lumina AutoML Banner](https://img.shields.io/badge/Lumina-AutoML-blueviolet?style=for-the-badge&logo=python&logoColor=white)

**A Beautiful, Intuitive Machine Learning Pipeline**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Ayushgirigoswami/Lumina-AutoML?style=flat-square&logo=github)](https://github.com/Ayushgirigoswami/Lumina-AutoML/stargazers)
[![Forks](https://img.shields.io/github/forks/Ayushgirigoswami/Lumina-AutoML?style=flat-square&logo=github)](https://github.com/Ayushgirigoswami/Lumina-AutoML/network/members)
[![Issues](https://img.shields.io/github/issues/Ayushgirigoswami/Lumina-AutoML?style=flat-square&logo=github)](https://github.com/Ayushgirigoswami/Lumina-AutoML/issues)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

---

### 🎯 Transform Your Data Into Insights With Just A Few Clicks

</div>

<div align="center">
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&pause=1000&color=6C63FF&center=true&vCenter=true&width=800&lines=Beautiful+CLI+Interface;Automated+ML+Pipeline;Smart+Feature+Engineering;Production-Ready+Models;Comprehensive+Visualizations" alt="Typing SVG" />
</div>

---
---
 <img width="1474" height="463" alt="Screenshot 2025-11-01 225620" src="https://github.com/user-attachments/assets/5bf3d945-33e6-4ba3-b721-683245d2138b" />

---

## 🎭 What is Lumina AutoML?

**Lumina AutoML** is not just another ML tool—it's your **intelligent companion** for building machine learning models. With a stunning command-line interface powered by Rich, it guides you through every step of the ML journey, from data loading to model deployment.

### 🌈 Why Lumina?

<table>
<tr>
<td width="50%">

#### 🎨 **Beautiful by Design**
- Stunning CLI with colors, progress bars, and animations
- Intuitive step-by-step workflow
- Clear visual feedback at every stage

#### 🧠 **Intelligent Automation**
- Auto-detects feature types
- Smart task type recommendation
- Handles missing values automatically
- Built-in feature scaling and encoding

</td>
<td width="50%">

#### 🚀 **Production Ready**
- Sklearn Pipeline integration
- Saves models in joblib format
- Comprehensive evaluation metrics
- Beautiful visualizations

#### ⚡ **Powerful Yet Simple**
- Support for 6+ algorithms
- Both classification & regression
- Real-time prediction mode
- Export cleaned datasets

</td>
</tr>
</table>

---

## ✨ Features

<div align="center">

| Feature | Description | Status |
|---------|-------------|--------|
| 📊 **Multi-Format Support** | CSV, Excel, JSON data loading | ✅ |
| 🔄 **Auto Preprocessing** | Handles missing values, scaling, encoding | ✅ |
| 🤖 **Multiple Algorithms** | RF, XGBoost, LightGBM, SVM, Linear models | ✅ |
| 📈 **Rich Visualizations** | Confusion matrix, feature importance, residuals | ✅ |
| 💾 **Model Persistence** | Save/load trained pipelines | ✅ |
| 🎯 **Smart Recommendations** | Auto-suggests classification vs regression | ✅ |
| 📋 **Detailed Reports** | Comprehensive evaluation metrics | ✅ |
| 🔮 **Prediction Mode** | Score new data with trained models | ✅ |

</div>

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Ayushgirigoswami/Lumina-AutoML.git
cd lumina-ml-cli

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 📦 Required Packages

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
rich>=10.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
openpyxl>=3.0.0
```

---

## 🎮 Quick Start

### 🌟 Training Mode

Launch the interactive ML pipeline:

```bash
python lumina_automl.py
```

**That's it!** Lumina will guide you through:

1. **📁 Data Loading** - Drag & drop your dataset
2. **🎯 Target Selection** - Pick what you want to predict
3. **🔧 Preprocessing** - Automatic data cleaning
4. **🤖 Model Training** - Choose from 6+ algorithms
5. **📊 Evaluation** - View metrics & visualizations
6. **💾 Results** - Save models & reports

### 🔮 Prediction Mode

Use a trained model on new data:

```bash
python lumina_automl.py --predict
```

---

## 📖 Documentation

### 🎯 Supported Algorithms

<details>
<summary><b>Classification Models</b></summary>

- **Random Forest** - Ensemble method, handles non-linearity
- **Logistic Regression** - Simple, interpretable baseline
- **Support Vector Machine** - Great for high-dimensional data
- **XGBoost** - Competition-winning gradient boosting
- **LightGBM** - Fast, handles large datasets efficiently

</details>

<details>
<summary><b>Regression Models</b></summary>

- **Random Forest** - Robust to outliers, non-linear
- **Linear Regression** - Simple baseline model
- **Ridge Regression** - Handles multicollinearity
- **Support Vector Regression** - Effective in high dimensions
- **XGBoost** - State-of-the-art gradient boosting
- **LightGBM** - Efficient for large-scale data

</details>

### 📊 Output Structure

```
ml_output_YYYYMMDD_HHMMSS/
│
├── models/
│   └── {model_name}_pipeline.joblib    # Trained pipeline
│
├── reports/
│   ├── evaluation_results.json         # Metrics & metadata
│   └── predictions.csv                 # Predictions (predict mode)
│
├── visualizations/
│   ├── confusion_matrix.png            # Classification only
│   ├── feature_importance.png          # If available
│   ├── actual_vs_predicted.png         # Regression only
│   └── residual_plot.png               # Regression only
│
└── cleaned_data/
    └── {filename}_cleaned_YYYYMMDD.csv # Optional export
```

---

## 💡 Examples

### Example 1: Credit Card Fraud Detection

```bash
# Launch Lumina
python lumina_automl.py

# Follow the prompts:
# 1. Load: fraud_detection.csv
# 2. Target: "is_fraud" (binary classification)
# 3. Model: Random Forest
# 4. Evaluate: View confusion matrix & metrics
```

**Output:**
- Accuracy: 98.5%
- F1 Score: 0.92
- Saved model ready for production

### Example 2: House Price Prediction

```bash
# Launch Lumina
python lumina_automl.py

# Follow the prompts:
# 1. Load: housing_data.csv
# 2. Target: "price" (regression)
# 3. Model: XGBoost
# 4. Evaluate: View R² score & residuals
```

**Output:**
- R² Score: 0.87
- RMSE: $15,432
- Feature importance chart

### Example 3: Batch Predictions

```bash
# Use trained model on new data
python lumina_automl.py --predict

# Follow the prompts:
# 1. Load saved pipeline
# 2. Load new data
# 3. Get predictions instantly
```

---

## 🛠️ Advanced Usage

### Custom Configuration

```python
from lumina_automl import ModernMLPipeline

# Create pipeline instance
pipeline = ModernMLPipeline()

# Load data programmatically
pipeline.data = pd.read_csv("your_data.csv")

# Set target
pipeline.target_column = "target"
pipeline.task_type = "classification"

# Train specific model
pipeline.preprocess_data()
pipeline.model_name = "XGBoost"
pipeline.train_model()
```

### Integration with Other Tools

```python
import joblib

# Load trained pipeline
model = joblib.load("ml_output_*/models/model.joblib")

# Use in your application
predictions = model.predict(new_data)
```

---

## 🤝 Contributing

We love contributions! Here's how you can help:

<div align="center">

| Type | How to Contribute |
|------|------------------|
| 🐛 **Bug Reports** | [Open an issue](https://github.com/Ayushgirigoswami/Lumina-AutoML/issues) with details |
| 💡 **Feature Requests** | Share your ideas in [discussions](https://github.com/Ayushgirigoswami/Lumina-AutoML/discussions) |
| 🔧 **Code** | Fork, code, test, and submit a PR |
| 📖 **Documentation** | Help improve docs and examples |
| ⭐ **Spread the Word** | Star the repo, share with friends! |

</div>

### Development Setup

```bash
# Fork and clone your fork
git clone  https://github.com/Ayushgirigoswami/Lumina-AutoML.git


# Create a feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 🏆 Acknowledgments

<div align="center">

Built with ❤️ using:

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Rich](https://img.shields.io/badge/Rich-009485?style=for-the-badge&logo=python&logoColor=white)](https://rich.readthedocs.io/)

</div>

Special thanks to:
- The **Rich** library for the beautiful terminal UI
- The **Scikit-learn** team for robust ML tools
- All contributors and supporters of this project

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

<div align="center">

### Need Help?

[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-red?style=for-the-badge&logo=github)](https://github.com/Ayushgirigoswami/Lumina-AutoML/issues)
[![Discussions](https://img.shields.io/badge/GitHub-Discussions-purple?style=for-the-badge&logo=github)](https://github.com/Ayushgirigoswami/Lumina-AutoML/discussions)

### Show Your Support

If you find Lumina AutoML helpful, please consider:

⭐ **Starring** the repository
🔗 **Sharing** with your network
💬 **Providing feedback** to help us improve

---

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=20&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=Happy+Machine+Learning!;Made+with+%E2%9D%A4%EF%B8%8F+by+the+Community;Star+%E2%AD%90+if+you+find+it+useful!" alt="Footer" />

**[⬆ Back to Top](#-lumina-automl)**

</div>

---

<div align="center">

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Ayushgirigoswami/Lumina-AutoML&type=Date)](https://star-history.com/#Ayushgirigoswami/Lumina-AutoML&Date)

</div>
