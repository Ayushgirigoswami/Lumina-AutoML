# 🌟 Lumina AutoML

<div align="center">

![Lumina AutoML Banner](https://raw.githubusercontent.com/Ayushgirigoswami/Lumina-AutoML/main/assets/banner.png)

**A Beautiful, User-Friendly Machine Learning Pipeline**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Ayushgirigoswami/Lumina-AutoML?style=social)](https://github.com/Ayushgirigoswami/Lumina-AutoML/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Ayushgirigoswami/Lumina-AutoML?style=social)](https://github.com/Ayushgirigoswami/Lumina-AutoML/network/members)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📖 About

**Lumina AutoML** is a modern, intuitive command-line tool that makes machine learning accessible to everyone. With its beautiful interface and step-by-step guidance, building powerful ML models has never been easier. Whether you're a beginner or an experienced data scientist, Lumina AutoML streamlines your workflow and helps you focus on what matters most - insights and results.

### Why Lumina AutoML?

- 🎨 **Beautiful CLI Interface** - Rich, colorful, and intuitive terminal UI
- 🚀 **Zero to Hero** - Go from raw data to trained model in minutes
- 🤖 **Automated Pipeline** - Handles preprocessing, training, and evaluation automatically
- 📊 **Visual Insights** - Generate beautiful plots and visualizations
- 🎯 **Smart Recommendations** - Get intelligent suggestions for task types and models
- 💾 **Export Ready** - Save cleaned data, models, and reports with one click

---

## ✨ Features

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Intuitive Workflow** | Step-by-step guidance through the entire ML pipeline |
| **Multi-Format Support** | Load CSV, Excel, and JSON files seamlessly |
| **Auto Preprocessing** | Automatic handling of missing values, encoding, and scaling |
| **Smart Task Detection** | Automatically identifies classification vs regression tasks |
| **Multiple Algorithms** | Support for Random Forest, XGBoost, LightGBM, SVM, and more |
| **Comprehensive Evaluation** | Detailed metrics, confusion matrices, and performance plots |
| **Export Functionality** | Save cleaned data in multiple formats (CSV, Excel, JSON) |
| **Model Persistence** | Save and reuse trained models |

### 🤖 Supported Algorithms

#### Classification
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier
- LightGBM Classifier

#### Regression
- Random Forest Regressor
- Linear Regression
- Ridge Regression
- Support Vector Regression (SVR)
- XGBoost Regressor
- LightGBM Regressor

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Ayushgirigoswami/Lumina-AutoML.git

# Navigate to the project directory
cd Lumina-AutoML

# Install required dependencies
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with the following dependencies:

```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
rich>=10.0.0
joblib>=1.1.0
openpyxl>=3.0.0
```

---

## 🎮 Quick Start

### Basic Usage

```bash
# Run the pipeline
python lumina_automl.py
```

### Step-by-Step Guide

#### 1️⃣ **Welcome Screen**
You'll be greeted with a beautiful ASCII art banner and feature overview.

#### 2️⃣ **Load Your Data**
```
Enter the path to your dataset (CSV, Excel, or JSON)
> /path/to/your/data.csv
```

#### 3️⃣ **Select Target Variable**
The tool will display all columns with statistics and help you choose your target variable.

#### 4️⃣ **Automatic Preprocessing**
Lumina AutoML handles:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Train-test splitting

Option to export cleaned data in your preferred format!

#### 5️⃣ **Model Selection**
Choose from multiple algorithms with detailed descriptions and recommendations.

#### 6️⃣ **Training & Evaluation**
Watch as your model trains with a beautiful progress bar, then view comprehensive metrics and visualizations.

---

## 📊 Example Workflow

### Sample Dataset: Iris Classification

```bash
$ python lumina_automl.py

# Step 1: Load Data
Enter the path to your dataset: iris.csv
✓ Successfully loaded dataset with 150 rows and 5 columns

# Step 2: Select Target
Enter the ID of your target column: 5
✓ Selected target: species
✓ Task type set to: classification

# Step 3: Preprocessing
✓ Preprocessing complete!
   - Numerical features: 4
   - Categorical features: 0
   - Training samples: 120
   - Testing samples: 30

# Step 4: Model Selection
Select a model:
1. Random Forest
2. Logistic Regression
3. XGBoost
Enter your choice: 1

# Step 5: Evaluation
Accuracy: 0.9667
Precision: 0.9667
Recall: 0.9667
F1 Score: 0.9667
```

### Output Structure

```
ml_output_20240101_120000/
├── models/
│   └── random_forest.joblib
├── reports/
│   └── evaluation_results.json
├── visualizations/
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── cleaned_data/
    └── iris_cleaned_20240101_120000.csv
```

---

## 📸 Screenshots

<div align="center">

### Welcome Screen
![Welcome Screen](https://via.placeholder.com/800x400/1e1e1e/00d4ff?text=Lumina+AutoML+Welcome+Screen)

### Data Loading
![Data Loading](https://via.placeholder.com/800x400/1e1e1e/00d4ff?text=Beautiful+Data+Preview+Table)

### Model Training
![Model Training](https://via.placeholder.com/800x400/1e1e1e/00d4ff?text=Progress+Bar+Animation)

### Results Dashboard
![Results](https://via.placeholder.com/800x400/1e1e1e/00d4ff?text=Comprehensive+Metrics+Display)

</div>

---

## 📚 Documentation

### Command Line Options

Currently, Lumina AutoML uses an interactive CLI. Future versions will support:
- `--config` - Load configuration from file
- `--auto` - Fully automated mode
- `--verbose` - Detailed logging

### Configuration

You can customize default behaviors by modifying the following variables in `lumina_automl.py`:

```python
OUTPUT_DIR = Path("ml_output_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
VISUALIZATIONS_DIR = Path("visualizations")
```

### API Reference

#### Class: `ModernMLPipeline`

**Methods:**
- `load_data()` - Load and preview dataset
- `select_target()` - Choose target variable and task type
- `preprocess_data()` - Handle missing values, encoding, and scaling
- `train_model()` - Train selected ML model
- `evaluate_model()` - Generate metrics and visualizations
- `show_summary()` - Display final results and next steps

---

## 🎨 Customization

### Color Scheme

Modify the `COLORS` dictionary to change the interface colors:

```python
COLORS = {
    "primary": "bright_blue",
    "secondary": "magenta",
    "accent": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "bright_white",
    "muted": "dim white"
}
```

### Model Parameters

Customize model hyperparameters in the `train_model()` method:

```python
"Random Forest": RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
```

---

## 🤝 Contributing

We love contributions! Here's how you can help make Lumina AutoML even better:

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Ayushgirigoswami/Lumina-AutoML.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests if applicable

4. **Commit Your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

5. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**

### Contribution Ideas

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧪 Additional test coverage
- 🌍 Internationalization

---

## 🗺️ Roadmap

### Version 2.0 (Coming Soon)
- [ ] Hyperparameter tuning with Grid/Random Search
- [ ] Cross-validation support
- [ ] Feature selection algorithms
- [ ] Model comparison mode
- [ ] API deployment guide
- [ ] Docker containerization

### Version 3.0 (Future)
- [ ] Web-based UI
- [ ] Real-time model monitoring
- [ ] AutoML with neural architecture search
- [ ] Time series forecasting
- [ ] Natural language processing tasks
- [ ] Computer vision support

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Ayush Giri Goswami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author

**Ayush Giri Goswami**

- GitHub: [@Ayushgirigoswami](https://github.com/Ayushgirigoswami)
- LinkedIn: [Connect with me](https://linkedin.com/in/ayushgirigoswami)
- Email: [Contact](mailto:ayushgirigoswami@example.com)

---

## 🙏 Acknowledgments

- **Rich** - For the beautiful terminal UI library
- **scikit-learn** - For the comprehensive ML algorithms
- **XGBoost & LightGBM** - For powerful gradient boosting implementations
- **Community** - For feedback and contributions

---

## 📊 Project Stats

<div align="center">

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Ayushgirigoswami/Lumina-AutoML)
![GitHub last commit](https://img.shields.io/github/last-commit/Ayushgirigoswami/Lumina-AutoML)
![GitHub code size](https://img.shields.io/github/languages/code-size/Ayushgirigoswami/Lumina-AutoML)

</div>

---

## ❓ FAQ

<details>
<summary><strong>Q: What Python version is required?</strong></summary>
<br>
Python 3.8 or higher is required. We recommend using Python 3.9+ for the best experience.
</details>

<details>
<summary><strong>Q: Can I use this for production deployments?</strong></summary>
<br>
Lumina AutoML is designed for rapid prototyping and learning. For production deployments, we recommend additional validation, monitoring, and deployment best practices.
</details>

<details>
<summary><strong>Q: How do I handle large datasets?</strong></summary>
<br>
For datasets larger than 1GB, consider using sampling or chunking techniques. Future versions will include better support for large-scale data.
</details>

<details>
<summary><strong>Q: Can I customize the models?</strong></summary>
<br>
Yes! You can modify the model parameters in the source code. We're working on a configuration file system for easier customization.
</details>

<details>
<summary><strong>Q: Is GPU support available?</strong></summary>
<br>
Currently, GPU support depends on your XGBoost and LightGBM installations. Future versions will include better GPU integration.
</details>

---

## 🌟 Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Ayushgirigoswami/Lumina-AutoML&type=Date)](https://star-history.com/#Ayushgirigoswami/Lumina-AutoML&Date)

</div>

---

## 💬 Support

Need help? Here's how to get support:

- 📖 Check the [Documentation](#-documentation)
- 💬 Start a [Discussion](https://github.com/Ayushgirigoswami/Lumina-AutoML/discussions)
- 🐛 Report a [Bug](https://github.com/Ayushgirigoswami/Lumina-AutoML/issues)
- ⭐ Star the repo if you find it helpful!

---

<div align="center">

**Made with ❤️ by Ayush Giri Goswami**

If you found this project helpful, please consider giving it a ⭐!

[⬆ Back to Top](#-lumina-automl)

</div>