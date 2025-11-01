#!/usr/bin/env python3
"""
Modern ML CLI - A beautiful, user-friendly machine learning pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Union

# Rich imports for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.tree import Tree
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich.live import Live

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           r2_score, mean_squared_error, mean_absolute_error, classification_report)
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

# Output directory setup
OUTPUT_DIR = Path("ml_output_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
VISUALIZATIONS_DIR = Path("visualizations")

# Color scheme
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

class ModernMLPipeline:
    """Modern Machine Learning Pipeline with a beautiful interface"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.target_column = None
        self.task_type = None  # 'classification' or 'regression'
        self.model = None
        self.model_name = None
        self.scaler = None
        self.imputer = None
        self.encoders = {}
        self.metrics = {}
        self.output_dir = OUTPUT_DIR
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for output"""
        for directory in [self.output_dir, 
                         self.output_dir / MODELS_DIR, 
                         self.output_dir / REPORTS_DIR,
                         self.output_dir / VISUALIZATIONS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def show_welcome(self):
        """Display a beautiful welcome screen"""
        console.clear()
        
        # ASCII Art Banner
        banner = """
 ██▓     █    ██  ███▄ ▄███▓ ██▓ ███▄    █  ▄▄▄          ▄▄▄       █    ██ ▄▄▄█████▓ ▒█████   ███▄ ▄███▓ ██▓    
▓██▒     ██  ▓██▒▓██▒▀█▀ ██▒▓██▒ ██ ▀█   █ ▒████▄       ▒████▄     ██  ▓██▒▓  ██▒ ▓▒▒██▒  ██▒▓██▒▀█▀ ██▒▓██▒    
▒██░    ▓██  ▒██░▓██    ▓██░▒██▒▓██  ▀█ ██▒▒██  ▀█▄     ▒██  ▀█▄  ▓██  ▒██░▒ ▓██░ ▒░▒██░  ██▒▓██    ▓██░▒██░    
▒██░    ▓▓█  ░██░▒██    ▒██ ░██░▓██▒  ▐▌██▒░██▄▄▄▄██    ░██▄▄▄▄██ ▓▓█  ░██░░ ▓██▓ ░ ▒██   ██░▒██    ▒██ ▒██░    
░██████▒▒▒█████▓ ▒██▒   ░██▒░██░▒██░   ▓██░ ▓█   ▓██▒    ▓█   ▓██▒▒▒█████▓   ▒██▒ ░ ░ ████▓▒░▒██▒   ░██▒░██████▒
░ ▒░▓  ░░▒▓▒ ▒ ▒ ░ ▒░   ░  ░░▓  ░ ▒░   ▒ ▒  ▒▒   ▓▒█░    ▒▒   ▓▒█░░▒▓▒ ▒ ▒   ▒ ░░   ░ ▒░▒░▒░ ░ ▒░   ░  ░░ ▒░▓  ░
░ ░ ▒  ░░░▒░ ░ ░ ░  ░      ░ ▒ ░░ ░░   ░ ▒░  ▒   ▒▒ ░     ▒   ▒▒ ░░░▒░ ░ ░     ░      ░ ▒ ▒░ ░  ░      ░░ ░ ▒  ░
  ░ ░    ░░░ ░ ░ ░      ░    ▒ ░   ░   ░ ░   ░   ▒        ░   ▒    ░░░ ░ ░   ░      ░ ░ ░ ▒  ░      ░     ░ ░   
    ░  ░   ░            ░    ░           ░       ░  ░         ░  ░   ░                  ░ ░         ░       ░  ░
                                                                                                                 
        """
        
        console.print(Text(banner, style=COLORS["primary"]))
        
        # Welcome message
        welcome_panel = Panel(
            Align.center(
                Text("Modern Machine Learning Pipeline", style=f"bold {COLORS['secondary']}") +
                Text("\n\nA beautiful, step-by-step approach to building ML models", style=COLORS["info"])
            ),
            border_style=COLORS["primary"],
            box=box.ROUNDED,
            title="[bold]Welcome[/bold]",
            title_align="center",
            padding=(1, 2)
        )
        
        features_panel = Panel(
            "\n".join([
                f"[{COLORS['success']}]✓[/{COLORS['success']}] [bold]Intuitive[/bold] step-by-step interface",
                f"[{COLORS['success']}]✓[/{COLORS['success']}] [bold]Automated[/bold] data preprocessing",
                f"[{COLORS['success']}]✓[/{COLORS['success']}] [bold]Smart[/bold] feature engineering",
                f"[{COLORS['success']}]✓[/{COLORS['success']}] [bold]Powerful[/bold] model training",
                f"[{COLORS['success']}]✓[/{COLORS['success']}] [bold]Beautiful[/bold] visualizations",
                f"[{COLORS['success']}]✓[/{COLORS['success']}] [bold]Comprehensive[/bold] model evaluation",
            ]),
            border_style=COLORS["accent"],
            box=box.ROUNDED,
            title="[bold]Features[/bold]",
            title_align="center"
        )
        
        console.print(Columns([welcome_panel, features_panel]))
        console.print("\n")
        
        # Start prompt
        console.print(f"[{COLORS['info']}]Press Enter to begin your ML journey...[/{COLORS['info']}]")
        input()
    
    def load_data(self) -> bool:
        """Step 1: Load data with a beautiful interface"""
        console.clear()
        
        step_panel = Panel(
            f"[{COLORS['info']}]We'll start by loading your dataset.[/{COLORS['info']}]",
            border_style=COLORS["primary"],
            title=f"[bold {COLORS['secondary']}]Step 1: Data Loading[/bold {COLORS['secondary']}]",
            title_align="left"
        )
        console.print(step_panel)
        
        # File path input with validation
        while True:
            file_path = Prompt.ask(
                f"[{COLORS['accent']}]Enter the path to your dataset[/{COLORS['accent']}] (CSV, Excel, or JSON)"
            )
            
            # Clean the file path
            file_path = file_path.strip('"\'').strip()
            file_path = os.path.expanduser(file_path)
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                console.print(f"[{COLORS['error']}]File not found: {file_path}[/{COLORS['error']}]")
                
                # Suggest files in the directory
                try:
                    parent_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
                    if os.path.exists(parent_dir):
                        files = [f for f in os.listdir(parent_dir) if f.endswith(('.csv', '.xlsx', '.xls', '.json'))]
                        if files:
                            console.print(f"[{COLORS['info']}]Available data files in this directory:[/{COLORS['info']}]")
                            for f in files[:5]:
                                console.print(f"  [{COLORS['accent']}]• {f}[/{COLORS['accent']}]")
                except Exception:
                    pass
                
                retry = Confirm.ask(f"[{COLORS['warning']}]Would you like to try again?[/{COLORS['warning']}]")
                if not retry:
                    return False
                continue
            
            # Try to load the file
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[{COLORS['info']}]{{task.description}}[/{COLORS['info']}]"),
                    BarColumn(complete_style=COLORS["primary"]),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Loading data...", total=100)
                    
                    # Simulate progress for better UX
                    for i in range(100):
                        time.sleep(0.01)
                        progress.update(task, advance=1)
                    
                    file_ext = Path(file_path).suffix.lower()
                    # Store the file path for later use
                    self.file_path = file_path
                    
                    if file_ext == '.csv':
                        self.data = pd.read_csv(file_path)
                    elif file_ext in ['.xlsx', '.xls']:
                        self.data = pd.read_excel(file_path)
                    elif file_ext == '.json':
                        self.data = pd.read_json(file_path)
                    else:
                        console.print(f"[{COLORS['error']}]Unsupported file format: {file_ext}[/{COLORS['error']}]")
                        continue
                
                # Show success message
                console.print(f"[{COLORS['success']}]✓ Successfully loaded dataset with {len(self.data)} rows and {len(self.data.columns)} columns[/{COLORS['success']}]")
                
                # Show data preview
                console.print(f"\n[bold {COLORS['secondary']}]Data Preview:[/bold {COLORS['secondary']}]")
                
                # Create a stylish table for data preview
                table = Table(
                    show_header=True,
                    header_style=f"bold {COLORS['primary']}",
                    box=box.ROUNDED
                )
                
                # Add columns
                for column in self.data.columns:
                    table.add_column(column)
                
                # Add rows (first 5)
                for _, row in self.data.head(5).iterrows():
                    table.add_row(*[str(val) for val in row])
                
                console.print(table)
                
                # Data info
                console.print(f"\n[bold {COLORS['secondary']}]Data Information:[/bold {COLORS['secondary']}]")
                
                # Create info table
                info_table = Table(
                    show_header=True,
                    header_style=f"bold {COLORS['primary']}",
                    box=box.ROUNDED
                )
                
                info_table.add_column("Column")
                info_table.add_column("Type")
                info_table.add_column("Non-Null Count")
                info_table.add_column("Missing Values")
                
                for col in self.data.columns:
                    dtype = str(self.data[col].dtype)
                    non_null = self.data[col].count()
                    missing = self.data[col].isna().sum()
                    missing_pct = f"{(missing / len(self.data)) * 100:.1f}%"
                    
                    # Color code for missing values
                    if missing > 0:
                        missing_text = f"{missing} ({missing_pct})"
                        missing_style = COLORS["warning"] if missing / len(self.data) < 0.2 else COLORS["error"]
                    else:
                        missing_text = "0 (0.0%)"
                        missing_style = COLORS["success"]
                    
                    info_table.add_row(
                        col,
                        dtype,
                        f"{non_null} / {len(self.data)}",
                        Text(missing_text, style=missing_style)
                    )
                
                console.print(info_table)
                
                # Continue prompt
                console.print(f"\n[{COLORS['info']}]Press Enter to continue to the next step...[/{COLORS['info']}]")
                input()
                return True
                
            except Exception as e:
                console.print(f"[{COLORS['error']}]Error loading file: {str(e)}[/{COLORS['error']}]")
                retry = Confirm.ask(f"[{COLORS['warning']}]Would you like to try again?[/{COLORS['warning']}]")
                if not retry:
                    return False
    
    def select_target(self) -> bool:
        """Step 2: Select target variable and determine task type"""
        console.clear()
        
        step_panel = Panel(
            f"[{COLORS['info']}]Now, let's select the target variable you want to predict.[/{COLORS['info']}]",
            border_style=COLORS["primary"],
            title=f"[bold {COLORS['secondary']}]Step 2: Target Selection[/bold {COLORS['secondary']}]",
            title_align="left"
        )
        console.print(step_panel)
        
        # Create a table of columns to choose from
        columns_table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            box=box.ROUNDED
        )
        
        columns_table.add_column("ID")
        columns_table.add_column("Column Name")
        columns_table.add_column("Type")
        columns_table.add_column("Unique Values")
        columns_table.add_column("Example Values")
        
        for i, col in enumerate(self.data.columns):
            unique_count = self.data[col].nunique()
            examples = str(self.data[col].dropna().sample(min(3, len(self.data))).tolist())
            if len(examples) > 30:
                examples = examples[:27] + "..."
                
            columns_table.add_row(
                str(i+1),
                col,
                str(self.data[col].dtype),
                str(unique_count),
                examples
            )
        
        console.print(columns_table)
        
        # Target selection
        while True:
            target_id = IntPrompt.ask(
                f"[{COLORS['accent']}]Enter the ID of your target column[/{COLORS['accent']}]",
                default=len(self.data.columns)
            )
            
            if target_id < 1 or target_id > len(self.data.columns):
                console.print(f"[{COLORS['error']}]Invalid ID. Please enter a number between 1 and {len(self.data.columns)}[/{COLORS['error']}]")
                continue
            
            self.target_column = self.data.columns[target_id-1]
            console.print(f"[{COLORS['success']}]✓ Selected target: {self.target_column}[/{COLORS['success']}]")
            
            # Determine task type based on data type and unique values
            target_data = self.data[self.target_column]
            unique_values = target_data.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            
            # Better task type detection logic with prominent suggestion
            console.print(f"\n[bold {COLORS['secondary']}]ML Task Recommendation:[/bold {COLORS['secondary']}]")
            
            suggestion_panel = None
            if is_numeric:
                # Check if the values are integers and few unique values (likely classification)
                if target_data.dropna().apply(lambda x: float(x).is_integer()).all() and unique_values <= 15:
                    suggested_task = "classification"
                    suggestion_panel = Panel(
                        f"Based on analysis, this target has {unique_values} unique integer values.\n\n"
                        f"[bold {COLORS['success']}]RECOMMENDED: CLASSIFICATION TASK[/bold {COLORS['success']}]\n\n"
                        f"This is ideal for predicting categories or classes.",
                        border_style=COLORS["success"],
                        title=f"[bold {COLORS['secondary']}]Task Suggestion[/bold {COLORS['secondary']}]",
                        title_align="center"
                    )
                else:
                    suggested_task = "regression"
                    suggestion_panel = Panel(
                        f"Based on analysis, this target has continuous numeric values.\n\n"
                        f"[bold {COLORS['success']}]RECOMMENDED: REGRESSION TASK[/bold {COLORS['success']}]\n\n"
                        f"This is ideal for predicting continuous values like prices, temperatures, etc.",
                        border_style=COLORS["success"],
                        title=f"[bold {COLORS['secondary']}]Task Suggestion[/bold {COLORS['secondary']}]",
                        title_align="center"
                    )
            else:
                # Non-numeric data is almost always classification
                suggested_task = "classification"
                suggestion_panel = Panel(
                    f"Based on analysis, this target has non-numeric categorical values.\n\n"
                    f"[bold {COLORS['success']}]RECOMMENDED: CLASSIFICATION TASK[/bold {COLORS['success']}]\n\n"
                    f"This is ideal for predicting categories or classes.",
                    border_style=COLORS["success"],
                    title=f"[bold {COLORS['secondary']}]Task Suggestion[/bold {COLORS['secondary']}]",
                    title_align="center"
                )
            
            console.print(suggestion_panel)
            
            # Show target data distribution for better decision making
            console.print(f"\n[bold {COLORS['secondary']}]Target Distribution:[/bold {COLORS['secondary']}]")
            
            if is_numeric and suggested_task == "regression":
                # For regression, show statistics
                stats = target_data.describe()
                
                stats_table = Table(
                    show_header=True,
                    header_style=f"bold {COLORS['primary']}",
                    box=box.ROUNDED
                )
                
                stats_table.add_column("Statistic")
                stats_table.add_column("Value")
                
                for stat, value in stats.items():
                    stats_table.add_row(stat, f"{value:.4f}" if isinstance(value, float) else str(value))
                
                console.print(stats_table)
            else:
                # For classification, show class distribution
                value_counts = target_data.value_counts().head(10)
                
                dist_table = Table(
                    show_header=True,
                    header_style=f"bold {COLORS['primary']}",
                    box=box.ROUNDED
                )
                
                dist_table.add_column("Class")
                dist_table.add_column("Count")
                dist_table.add_column("Percentage")
                
                for value, count in value_counts.items():
                    percentage = (count / len(target_data)) * 100
                    dist_table.add_row(
                        str(value),
                        str(count),
                        f"{percentage:.2f}%"
                    )
                
                console.print(dist_table)
            
            # Let user confirm or change the task type using a numbered menu
            task_options = ["classification", "regression"]
            task_index = task_options.index(suggested_task)
            
            # Create a task selection table
            task_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                box=box.ROUNDED
            )
            
            task_table.add_column("Option")
            task_table.add_column("Task Type")
            task_table.add_column("Description")
            
            task_table.add_row(
                "1", 
                "Classification", 
                "Predict categories or classes (e.g., spam/not spam, customer segments)"
            )
            task_table.add_row(
                "2", 
                "Regression", 
                "Predict continuous values (e.g., house prices, temperature)"
            )
            
            console.print(task_table)
            
            # Use IntPrompt for numeric selection to avoid spelling errors
            default_option = 1 if suggested_task == "classification" else 2
            task_option = IntPrompt.ask(
                f"[{COLORS['accent']}]Select task type (1-2)[/{COLORS['accent']}]",
                default=default_option
            )
            
            # Convert numeric choice to task type
            if task_option == 1:
                self.task_type = "classification"
            else:
                self.task_type = "regression"
                
            console.print(f"[{COLORS['success']}]✓ Task type set to: {self.task_type}[/{COLORS['success']}]")
            
            # Continue prompt
            console.print(f"\n[{COLORS['info']}]Press Enter to continue to the next step...[/{COLORS['info']}]")
            input()
            return True
    
    def preprocess_data(self) -> bool:
        """Step 3: Data preprocessing"""
        console.clear()
        
        step_panel = Panel(
            f"[{COLORS['info']}]Let's preprocess the data to prepare it for modeling.[/{COLORS['info']}]",
            border_style=COLORS["primary"],
            title=f"[bold {COLORS['secondary']}]Step 3: Data Preprocessing[/bold {COLORS['secondary']}]",
            title_align="left"
        )
        console.print(step_panel)
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[{COLORS['info']}]{{task.description}}[/{COLORS['info']}]"),
            BarColumn(complete_style=COLORS["primary"]),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # 1. Separate features and target
            task1 = progress.add_task("Separating features and target...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task1, advance=1)
            
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            # 2. Identify numerical and categorical features
            task2 = progress.add_task("Identifying feature types...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task2, advance=1)
            
            self.feature_names = X.columns.tolist()
            self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            # 3. Handle missing values
            task3 = progress.add_task("Handling missing values...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task3, advance=1)
            
            # For numerical features
            if self.numerical_features:
                self.imputer = SimpleImputer(strategy='median')
                X[self.numerical_features] = self.imputer.fit_transform(X[self.numerical_features])
            
            # For categorical features
            for col in self.categorical_features:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
            
            # 4. Encode categorical features
            task4 = progress.add_task("Encoding categorical features...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task4, advance=1)
            
            for col in self.categorical_features:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                self.encoders[col] = encoder
            
            # 5. Scale numerical features
            task5 = progress.add_task("Scaling numerical features...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task5, advance=1)
            
            if self.numerical_features:
                self.scaler = StandardScaler()
                X[self.numerical_features] = self.scaler.fit_transform(X[self.numerical_features])
            
            # 6. Split data
            task6 = progress.add_task("Splitting data into train and test sets...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task6, advance=1)
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Show preprocessing summary
        console.print(f"\n[bold {COLORS['secondary']}]Preprocessing Summary:[/bold {COLORS['secondary']}]")
        
        summary_table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            box=box.ROUNDED
        )
        
        summary_table.add_column("Step")
        summary_table.add_column("Details")
        
        summary_table.add_row(
            "Feature Types",
            f"Numerical: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}"
        )
        summary_table.add_row(
            "Missing Values",
            f"Numerical: Imputed with median, Categorical: Filled with mode"
        )
        summary_table.add_row(
            "Encoding",
            f"Categorical features encoded using Label Encoding"
        )
        summary_table.add_row(
            "Scaling",
            f"Numerical features scaled using StandardScaler"
        )
        summary_table.add_row(
            "Data Split",
            f"Training: {len(self.X_train)} samples, Testing: {len(self.X_test)} samples"
        )
        
        console.print(summary_table)
        
        # Option to download cleaned data
        console.print(f"\n[bold {COLORS['secondary']}]Download Cleaned Data[/bold {COLORS['secondary']}]")
        
        download_cleaned = Confirm.ask(
            f"[{COLORS['accent']}]Would you like to download the cleaned dataset?[/{COLORS['accent']}]",
            default=True
        )
        
        if download_cleaned:
            # Create a DataFrame with the preprocessed data
            cleaned_data = pd.DataFrame(
                np.concatenate([self.X_train, self.X_test]), 
                columns=self.feature_names
            )
            # Add the target column back
            cleaned_data[self.target_column] = pd.concat([self.y_train, self.y_test]).values
            
            # Ask for file format
            console.print(f"\n[{COLORS['info']}]Select the file format for the cleaned data:[/{COLORS['info']}]")
            formats = ["CSV", "Excel", "JSON"]
            
            for i, fmt in enumerate(formats):
                console.print(f"[{COLORS['accent']}]{i+1}.[/{COLORS['accent']}] {fmt}")
            
            format_choice = IntPrompt.ask(
                f"[{COLORS['accent']}]Enter your choice (1-{len(formats)})[/{COLORS['accent']}]",
                default=1
            )
            
            # Create cleaned data directory if it doesn't exist
            cleaned_data_dir = self.output_dir / "cleaned_data"
            cleaned_data_dir.mkdir(exist_ok=True)
            
            # Generate filename based on original file
            original_filename = Path(self.file_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[{COLORS['info']}]{{task.description}}[/{COLORS['info']}]"),
                BarColumn(complete_style=COLORS["primary"]),
                console=console
            ) as progress:
                task = progress.add_task("Saving cleaned data...", total=100)
                
                # Simulate progress
                for i in range(100):
                    time.sleep(0.01)
                    progress.update(task, advance=1)
                
                # Save the cleaned data
                if format_choice == 1:  # CSV
                    cleaned_file_path = cleaned_data_dir / f"{original_filename}_cleaned_{timestamp}.csv"
                    cleaned_data.to_csv(cleaned_file_path, index=False)
                elif format_choice == 2:  # Excel
                    cleaned_file_path = cleaned_data_dir / f"{original_filename}_cleaned_{timestamp}.xlsx"
                    cleaned_data.to_excel(cleaned_file_path, index=False)
                else:  # JSON
                    cleaned_file_path = cleaned_data_dir / f"{original_filename}_cleaned_{timestamp}.json"
                    cleaned_data.to_json(cleaned_file_path, orient="records")
            
            console.print(f"[{COLORS['success']}]✓ Cleaned data saved to: {cleaned_file_path}[/{COLORS['success']}]")
        
        # Continue prompt
        console.print(f"\n[{COLORS['info']}]Press Enter to continue to the next step...[/{COLORS['info']}]")
        input()
        return True
    
    def train_model(self) -> bool:
        """Step 4: Model training"""
        console.clear()
        
        step_panel = Panel(
            f"[{COLORS['info']}]Now, let's train a machine learning model.[/{COLORS['info']}]",
            border_style=COLORS["primary"],
            title=f"[bold {COLORS['secondary']}]Step 4: Model Training[/bold {COLORS['secondary']}]",
            title_align="left"
        )
        console.print(step_panel)
        
        # Display current task type
        console.print(f"[bold {COLORS['info']}]Current task type: {self.task_type}[/bold {COLORS['info']}]")
        
        # Verify target variable is appropriate for the task type
        if self.task_type == "classification":
            # For classification, check if target is categorical or has few unique values
            unique_values = self.y_train.nunique()
            if unique_values > 15 and pd.api.types.is_numeric_dtype(self.y_train):
                console.print(f"[{COLORS['warning']}]Warning: Your target has {unique_values} unique values, which is unusual for classification.[/{COLORS['warning']}]")
                change_task = Confirm.ask(f"[{COLORS['warning']}]Would you like to switch to regression instead?[/{COLORS['warning']}]")
                if change_task:
                    self.task_type = "regression"
                    console.print(f"[{COLORS['success']}]✓ Task type changed to: regression[/{COLORS['success']}]")
        else:  # regression
            # For regression, check if target is numeric and has many unique values
            if not pd.api.types.is_numeric_dtype(self.y_train):
                console.print(f"[{COLORS['warning']}]Warning: Your target is not numeric, which is required for regression.[/{COLORS['warning']}]")
                change_task = Confirm.ask(f"[{COLORS['warning']}]Would you like to switch to classification instead?[/{COLORS['warning']}]")
                if change_task:
                    self.task_type = "classification"
                    console.print(f"[{COLORS['success']}]✓ Task type changed to: classification[/{COLORS['success']}]")
            elif self.y_train.nunique() <= 5:
                console.print(f"[{COLORS['warning']}]Warning: Your target has only {self.y_train.nunique()} unique values, which might be better suited for classification.[/{COLORS['warning']}]")
                change_task = Confirm.ask(f"[{COLORS['warning']}]Would you like to switch to classification instead?[/{COLORS['warning']}]")
                if change_task:
                    self.task_type = "classification"
                    console.print(f"[{COLORS['success']}]✓ Task type changed to: classification[/{COLORS['success']}]")
        
        # Model selection
        models = {
            "classification": {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Support Vector Machine": SVC(probability=True, random_state=42),
                "XGBoost": xgb.XGBClassifier(random_state=42),
                "LightGBM": lgb.LGBMClassifier(random_state=42)
            },
            "regression": {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=42),
                "Support Vector Regression": SVR(),
                "XGBoost": xgb.XGBRegressor(random_state=42),
                "LightGBM": lgb.LGBMRegressor(random_state=42)
            }
        }
        
        # Create a table of available models
        models_table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            box=box.ROUNDED
        )
        
        models_table.add_column("ID")
        models_table.add_column("Model")
        models_table.add_column("Description")
        models_table.add_column("Best For")
        
        model_descriptions = {
            "Random Forest": {
                "desc": "Ensemble method using multiple decision trees. Good for most tasks, handles non-linearity well.",
                "best_for": "Complex relationships, handles missing values well"
            },
            "Logistic Regression": {
                "desc": "Simple linear model for classification. Interpretable but may underfit complex data.",
                "best_for": "Binary/multiclass classification with linear boundaries"
            },
            "Linear Regression": {
                "desc": "Simple linear model for regression. Interpretable but may underfit complex data.",
                "best_for": "Simple continuous predictions with linear relationships"
            },
            "Ridge Regression": {
                "desc": "Linear regression with regularization. Helps with multicollinearity.",
                "best_for": "Regression with many correlated features"
            },
            "Support Vector Machine": {
                "desc": "Finds optimal hyperplane to separate classes. Good for high-dimensional data.",
                "best_for": "Classification with clear margins between classes"
            },
            "Support Vector Regression": {
                "desc": "SVR version for regression tasks. Good for high-dimensional data.",
                "best_for": "Regression with high-dimensional data"
            },
            "XGBoost": {
                "desc": "Gradient boosting implementation. Often wins competitions, handles various data types.",
                "best_for": "Complex tasks where performance is critical"
            },
            "LightGBM": {
                "desc": "Light gradient boosting framework. Fast training speed, handles large datasets well.",
                "best_for": "Large datasets with many samples"
            }
        }
        
        available_models = list(models[self.task_type].keys())
        for i, model_name in enumerate(available_models):
            models_table.add_row(
                str(i+1),
                model_name,
                model_descriptions.get(model_name, {}).get("desc", ""),
                model_descriptions.get(model_name, {}).get("best_for", "")
            )
        
        console.print(models_table)
        
        # Model selection
        while True:
            model_id = IntPrompt.ask(
                f"[{COLORS['accent']}]Enter the ID of the model you want to train[/{COLORS['accent']}]",
                default=1
            )
            
            if model_id < 1 or model_id > len(available_models):
                console.print(f"[{COLORS['error']}]Invalid ID. Please enter a number between 1 and {len(available_models)}[/{COLORS['error']}]")
                continue
            
            self.model_name = available_models[model_id-1]
            self.model = models[self.task_type][self.model_name]
            
            console.print(f"[{COLORS['success']}]✓ Selected model: {self.model_name}[/{COLORS['success']}]")
            break
        
        # Train the model with a progress animation and proper error handling
        console.print(f"\n[bold {COLORS['secondary']}]Training {self.model_name}...[/bold {COLORS['secondary']}]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[{COLORS['info']}]{{task.description}}[/{COLORS['info']}]"),
                BarColumn(complete_style=COLORS["primary"]),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                train_task = progress.add_task(f"Training {self.model_name}...", total=100)
                
                # Simulate initial progress for better UX
                for i in range(50):
                    time.sleep(0.02)
                    progress.update(train_task, advance=0.5)
                
                # Actually train the model
                self.model.fit(self.X_train, self.y_train)
                
                # Complete the progress bar
                for i in range(50, 100):
                    time.sleep(0.01)
                    progress.update(train_task, advance=1)
            
            console.print(f"[{COLORS['success']}]✓ Model training complete![/{COLORS['success']}]")
            
            # Save the model
            model_path = self.output_dir / MODELS_DIR / f"{self.model_name.replace(' ', '_').lower()}.joblib"
            joblib.dump(self.model, model_path)
            console.print(f"[{COLORS['info']}]Model saved to: {model_path}[/{COLORS['info']}]")
            
            # Continue prompt
            console.print(f"\n[{COLORS['info']}]Press Enter to continue to the next step...[/{COLORS['info']}]")
            input()
            return True
            
        except Exception as e:
            # Handle training errors gracefully
            error_panel = Panel(
                f"[{COLORS['error']}]{str(e)}[/{COLORS['error']}]",
                border_style=COLORS["error"],
                title=f"[bold {COLORS['error']}]Training Error[/bold {COLORS['error']}]",
                title_align="left"
            )
            console.print(error_panel)
            
            # Provide helpful suggestions based on the error
            if "Unknown label type: continuous" in str(e):
                console.print(f"[{COLORS['warning']}]It looks like you're trying to use a classification model on regression data.[/{COLORS['warning']}]")
                console.print(f"[{COLORS['info']}]Suggestion: Change the task type to 'regression' and select a regression model.[/{COLORS['info']}]")
                
                change_task = Confirm.ask(f"[{COLORS['accent']}]Would you like to switch to regression and try again?[/{COLORS['accent']}]")
                if change_task:
                    self.task_type = "regression"
                    console.print(f"[{COLORS['success']}]✓ Task type changed to: regression[/{COLORS['success']}]")
                    console.print(f"[{COLORS['info']}]Press Enter to return to model selection...[/{COLORS['info']}]")
                    input()
                    return self.train_model()  # Restart the training process
            
            elif "not convertible to float" in str(e) or "could not convert string to float" in str(e):
                console.print(f"[{COLORS['warning']}]It looks like your data contains non-numeric values that can't be processed by this model.[/{COLORS['warning']}]")
                console.print(f"[{COLORS['info']}]Suggestion: Make sure all features are properly encoded before training.[/{COLORS['info']}]")
            
            elif "check_array" in str(e) and "nan" in str(e).lower():
                console.print(f"[{COLORS['warning']}]Your data contains missing values (NaN) that weren't properly handled.[/{COLORS['warning']}]")
                console.print(f"[{COLORS['info']}]Suggestion: Use a different imputation strategy or remove rows with missing values.[/{COLORS['info']}]")
            
            else:
                console.print(f"[{COLORS['info']}]Suggestion: Try a different model that might be more compatible with your data.[/{COLORS['info']}]")
            
            retry = Confirm.ask(f"[{COLORS['accent']}]Would you like to select a different model?[/{COLORS['accent']}]")
            if retry:
                console.print(f"[{COLORS['info']}]Press Enter to return to model selection...[/{COLORS['info']}]")
                input()
                return self.train_model()  # Restart the training process
            
            return False
    
    def evaluate_model(self) -> bool:
        """Step 5: Model evaluation"""
        console.clear()
        
        step_panel = Panel(
            f"[{COLORS['info']}]Let's evaluate the model's performance.[/{COLORS['info']}]",
            border_style=COLORS["primary"],
            title=f"[bold {COLORS['secondary']}]Step 5: Model Evaluation[/bold {COLORS['secondary']}]",
            title_align="left"
        )
        console.print(step_panel)
        
        # Make predictions
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[{COLORS['info']}]{{task.description}}[/{COLORS['info']}]"),
            BarColumn(complete_style=COLORS["primary"]),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Making predictions...", total=100)
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.02)
                progress.update(task, advance=1)
            
            # Make predictions
            if self.task_type == "classification":
                y_pred = self.model.predict(self.X_test)
                y_prob = self.model.predict_proba(self.X_test) if hasattr(self.model, "predict_proba") else None
            else:
                y_pred = self.model.predict(self.X_test)
                y_prob = None
        
        # Calculate metrics
        if self.task_type == "classification":
            # Classification metrics
            self.metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Create a beautiful metrics display
            metrics_panel = Panel(
                Align.center(
                    Text(f"Accuracy: {self.metrics['accuracy']:.4f}", style=f"bold {COLORS['success']}") +
                    Text("\n\n") +
                    Text(f"Precision: {self.metrics['precision']:.4f}", style=COLORS["info"]) +
                    Text("\n") +
                    Text(f"Recall: {self.metrics['recall']:.4f}", style=COLORS["info"]) +
                    Text("\n") +
                    Text(f"F1 Score: {self.metrics['f1']:.4f}", style=COLORS["info"])
                ),
                border_style=COLORS["primary"],
                box=box.ROUNDED,
                title=f"[bold]Classification Metrics[/bold]",
                title_align="center",
                padding=(1, 2)
            )
            
            console.print(metrics_panel)
            
            # Classification report
            console.print(f"\n[bold {COLORS['secondary']}]Detailed Classification Report:[/bold {COLORS['secondary']}]")
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            report_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                box=box.ROUNDED
            )
            
            report_table.add_column("Class")
            report_table.add_column("Precision")
            report_table.add_column("Recall")
            report_table.add_column("F1-Score")
            report_table.add_column("Support")
            
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                report_table.add_row(
                    str(class_name),
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}",
                    str(metrics['support'])
                )
            
            console.print(report_table)
            
            # Create confusion matrix visualization
            plt.figure(figsize=(8, 6))
            cm = pd.crosstab(self.y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            cm_path = self.output_dir / VISUALIZATIONS_DIR / "confusion_matrix.png"
            plt.savefig(cm_path)
            console.print(f"[{COLORS['info']}]Confusion matrix saved to: {cm_path}[/{COLORS['info']}]")
            
        else:
            # Regression metrics
            self.metrics = {
                "r2": r2_score(self.y_test, y_pred),
                "mse": mean_squared_error(self.y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
                "mae": mean_absolute_error(self.y_test, y_pred)
            }
            
            # Create a beautiful metrics display
            metrics_panel = Panel(
                Align.center(
                    Text(f"R² Score: {self.metrics['r2']:.4f}", style=f"bold {COLORS['success']}") +
                    Text("\n\n") +
                    Text(f"Mean Squared Error: {self.metrics['mse']:.4f}", style=COLORS["info"]) +
                    Text("\n") +
                    Text(f"Root Mean Squared Error: {self.metrics['rmse']:.4f}", style=COLORS["info"]) +
                    Text("\n") +
                    Text(f"Mean Absolute Error: {self.metrics['mae']:.4f}", style=COLORS["info"])
                ),
                border_style=COLORS["primary"],
                box=box.ROUNDED,
                title=f"[bold]Regression Metrics[/bold]",
                title_align="center",
                padding=(1, 2)
            )
            
            console.print(metrics_panel)
            
            # Create scatter plot of actual vs predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            plt.tight_layout()
            
            scatter_path = self.output_dir / VISUALIZATIONS_DIR / "actual_vs_predicted.png"
            plt.savefig(scatter_path)
            console.print(f"[{COLORS['info']}]Scatter plot saved to: {scatter_path}[/{COLORS['info']}]")
            
            # Create residual plot
            plt.figure(figsize=(8, 6))
            residuals = self.y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.tight_layout()
            
            residual_path = self.output_dir / VISUALIZATIONS_DIR / "residual_plot.png"
            plt.savefig(residual_path)
            console.print(f"[{COLORS['info']}]Residual plot saved to: {residual_path}[/{COLORS['info']}]")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            console.print(f"\n[bold {COLORS['secondary']}]Feature Importance:[/bold {COLORS['secondary']}]")
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Create feature importance table
            importance_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                box=box.ROUNDED
            )
            
            importance_table.add_column("Feature")
            importance_table.add_column("Importance")
            
            for _, row in feature_importance.head(10).iterrows():
                importance_table.add_row(
                    row['Feature'],
                    f"{row['Importance']:.4f}"
                )
            
            console.print(importance_table)
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            
            importance_path = self.output_dir / VISUALIZATIONS_DIR / "feature_importance.png"
            plt.savefig(importance_path)
            console.print(f"[{COLORS['info']}]Feature importance plot saved to: {importance_path}[/{COLORS['info']}]")
        
        # Save evaluation results
        results = {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = self.output_dir / REPORTS_DIR / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        console.print(f"[{COLORS['info']}]Evaluation results saved to: {results_path}[/{COLORS['info']}]")
        
        # Continue prompt
        console.print(f"\n[{COLORS['info']}]Press Enter to continue to the final step...[/{COLORS['info']}]")
        input()
        return True
    
    def show_summary(self) -> None:
        """Step 6: Show summary and next steps"""
        console.clear()
        
        step_panel = Panel(
            f"[{COLORS['info']}]Congratulations! Your ML pipeline is complete.[/{COLORS['info']}]",
            border_style=COLORS["primary"],
            title=f"[bold {COLORS['secondary']}]Step 6: Summary & Next Steps[/bold {COLORS['secondary']}]",
            title_align="left"
        )
        console.print(step_panel)
        
        # Create a unified summary panel with clear sections
        summary_table = Table(show_header=False, box=None, padding=(0, 1))
        summary_table.add_column("Pipeline Steps", style=COLORS["success"])
        
        summary_table.add_row("✅ Data loaded and preprocessed")
        summary_table.add_row("✅ Target variable selected")
        summary_table.add_row("✅ Model trained successfully")
        summary_table.add_row("✅ Model evaluated and visualized")
        summary_table.add_row("✅ Results and artifacts saved")
        
        # Create metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column("Metrics", style=COLORS["info"])
        
        metrics_table.add_row(f"[bold {COLORS['secondary']}]Model: {self.model_name}[/bold {COLORS['secondary']}]")
        
        if self.task_type == "classification":
            metrics_table.add_row(f"Task: Classification")
            metrics_table.add_row(f"Accuracy: {self.metrics['accuracy']:.4f}")
            metrics_table.add_row(f"F1 Score: {self.metrics['f1']:.4f}")
        else:
            metrics_table.add_row(f"Task: Regression")
            metrics_table.add_row(f"R² Score: {self.metrics['r2']:.4f}")
            metrics_table.add_row(f"RMSE: {self.metrics['rmse']:.4f}")
        
        # Create a grid layout with two panels side by side
        grid = Table.grid(padding=2)
        grid.add_column("Summary")
        grid.add_column("Metrics")
        
        # Add the panels to the grid
        summary_panel = Panel(
            summary_table,
            border_style=COLORS["primary"],
            box=box.ROUNDED,
            title=f"[bold]Pipeline Summary[/bold]",
            title_align="center",
            width=40
        )
        
        metrics_panel = Panel(
            metrics_table,
            border_style=COLORS["primary"],
            box=box.ROUNDED,
            title=f"[bold]Model Performance[/bold]",
            title_align="center",
            width=40
        )
        
        grid.add_row(summary_panel, metrics_panel)
        console.print(grid)
        
        # Output directory information
        console.print(f"\n[bold {COLORS['secondary']}]Output Directory:[/bold {COLORS['secondary']}] {self.output_dir}")
        console.print(f"[{COLORS['info']}]• Models: {self.output_dir / MODELS_DIR}[/{COLORS['info']}]")
        console.print(f"[{COLORS['info']}]• Reports: {self.output_dir / REPORTS_DIR}[/{COLORS['info']}]")
        console.print(f"[{COLORS['info']}]• Visualizations: {self.output_dir / VISUALIZATIONS_DIR}[/{COLORS['info']}]")
        
        # Next steps
        next_steps_panel = Panel(
            "\n".join([
                f"[{COLORS['info']}]1. Explore the saved visualizations to gain insights[/{COLORS['info']}]",
                f"[{COLORS['info']}]2. Use the trained model for predictions on new data[/{COLORS['info']}]",
                f"[{COLORS['info']}]3. Try different models or hyperparameter tuning for better performance[/{COLORS['info']}]",
                f"[{COLORS['info']}]4. Deploy the model to a production environment[/{COLORS['info']}]"
            ]),
            border_style=COLORS["accent"],
            box=box.ROUNDED,
            title=f"[bold]Next Steps[/bold]",
            title_align="center"
        )
        
        console.print(next_steps_panel)
        
        # Farewell message
        console.print(f"\n[bold {COLORS['secondary']}]Thank you for using Modern ML Pipeline![/bold {COLORS['secondary']}]")
        console.print(f"[{COLORS['info']}]Run this tool again to create another ML pipeline.[/{COLORS['info']}]")

def main():
    """Main function to run the ML pipeline"""
    ml_pipeline = ModernMLPipeline()
    ml_pipeline.show_welcome()
    
    if ml_pipeline.load_data():
        if ml_pipeline.select_target():
            if ml_pipeline.preprocess_data():
                if ml_pipeline.train_model():
                    if ml_pipeline.evaluate_model():
                        ml_pipeline.show_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(f"\n[{COLORS['warning']}]Pipeline interrupted. Exiting...[/{COLORS['warning']}]")
    except Exception as e:
        console.print(f"\n[{COLORS['error']}]An error occurred: {str(e)}[/{COLORS['error']}]")
        console.print_exception()