# Lumina-AutoML
🌟 Lumina AutoML - Beautiful, automated machine learning in your terminal. Smart task detection, one-click preprocessing, multiple models (XGBoost, LightGBM, RF), comprehensive evaluation with visualizations. Transform raw data into trained models with an elegant, colorful CLI experience


## Features

- 🎨 **Modern, Colorful Interface**: Built with Rich library for a visually appealing CLI experience
- 🔄 **Step-by-Step ML Pipeline**: Guided workflow from data loading to model evaluation
- 📊 **Smart Task Detection**: Automatically suggests classification or regression based on data analysis
- 🧹 **Automated Data Preprocessing**: Handles missing values, encoding, and scaling
- 📈 **Multiple Model Options**: Supports various ML algorithms for both classification and regression
- 📉 **Comprehensive Evaluation**: Detailed metrics and visualizations for model performance
- 💾 **Download Cleaned Data**: Option to save preprocessed data in various formats
- 📁 **Organized Output**: Saves models, reports, and visualizations in a structured directory

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lumina-ml-cli.git
cd lumina-ml-cli
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the tool with:
```bash
python modern_ml_cli.py
```

Follow the interactive prompts to:
1. Load your dataset (CSV, Excel, or JSON)
2. Select a target variable
3. Preprocess your data (with option to download cleaned data)
4. Train a machine learning model
5. Evaluate model performance
6. View results and next steps

## Supported Models

### Classification
- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost
- LightGBM

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Support Vector Regression
- XGBoost
- LightGBM

## Example Output

The tool creates an organized output directory with:
- Trained model files (joblib format)
- Evaluation reports (JSON)
- Visualizations (confusion matrices, feature importance, etc.)
- Cleaned data (optional)

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
