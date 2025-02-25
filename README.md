# A/B Test Framework

![Preview of the framework](img/ab_test_framework.png)

## Overview
This framework provides a structured, end-to-end approach to conducting A/B tests, from experiment design to results analysis, including segmentation analysis. It is designed to be a flexible and reusable template for running A/B experiments efficiently.

## Problem Statement
A/B testing is a fundamental technique for data-driven decision-making, but implementing it correctly requires a structured approach. This framework provides a streamlined methodology to:
- Define experiment objectives and key metrics.
- Ensure proper sample size and distribution.
- Apply appropriate statistical techniques.
- Analyze results globally and by segments.

## Technologies
- **Python**: Core programming language.
- **pandas**: Data manipulation and cleaning.
- **scipy & statsmodels**: Statistical testing and hypothesis validation.
- **matplotlib & seaborn**: Data visualization.
- **Jupyter Notebooks**: Interactive experiment design and analysis.

## Project Structure

```bash
AB-TEST-FRAMEWORK/
│
├── data/                           # Example datasets for A/B tests
│   ├── ab_test_example_1.csv
│   ├── ab_test_example_2.csv
│   └── ab_test_example_3.csv
│
├── img/                            # Images for documentation
│
├── notebooks/                      # Notebooks for experiment execution
│   ├── templates/                  # Templates for standard A/B testing workflows
│   ├── ab-test_example_1.ipynb     # Real-world example 1
│   ├── ab-test_example_2.ipynb     # Real-world example 2
│   ├── ab-test_example_3.ipynb     # Real-world example 3
│   ├── ab-test-segments_example_1.ipynb  # Segmented analysis example 1
│   └── ab-test-segments_example_3.ipynb  # Segmented analysis example 3
│
├── src/                            # Core Python functions for A/B testing
│   ├── ab_tests.py                 # Functions for executing A/B tests
│   ├── analysis.py                 # Data cleaning and statistical analysis
│   └── framework.py                # Experiment setup and result aggregation
│
├── requirements.txt                # Dependencies (pandas, scikit-learn, scrapy, etc.)
├── environment.yml                 # Conda environment configuration
├── .gitignore                      # Ignore unnecessary files
├── LICENSE                         # License information
└── README.md                       # Project documentation
```

## Workflow
### 1. Experiment Design
The framework provides structured templates to design experiments:
- **Define experiment objectives**.
- **Select key metrics** (conversion rate, retention, revenue, etc.).
- **Determine required sample size**.
- **Ensure data quality** before analysis.

### 2. Data Analysis
- **Load and clean data**: Handle missing values, duplicates, and inconsistencies.
- **Check sample distribution**: Ensure fair representation across groups.
- **Validate sample variance**: Confirm homogeneity before statistical tests.

### 3. Statistical Testing
- **Choose the appropriate test** (e.g., t-test, chi-square, Mann-Whitney U).
- **Adjust for unbalanced samples** if necessary.
- **Perform hypothesis testing** to determine statistical significance.
- **Apply multiple comparison corrections** if required.

### 4. Segment Analysis (Optional)
- **Define segmentation criteria** (demographics, device type, location, etc.).
- **Perform A/B analysis per segment** to detect differential impacts.
- **Test for interactions** between variables to refine insights.

## Running the Framework
To execute an A/B test, open one of the template notebooks and follow the guided steps.

For segment analysis, use the **ab-test-segments_framework_template.ipynb** notebook to break down results by predefined groups.

## Example Usage
- Change the sources
- Adapt the columns in the data cleaning phase, inspect the inconsistencies and 
- Calculate your metrics.
- Select your columns: variant, metric and segment.
- Run the complete notebook, all the analysis and test will be automatically run, selecting the most appropiated test for your data distribution.

```python
from src.ab_tests import run_ab_test

data = load_data("data/ab_test_example_1.csv")
results = run_ab_test(data, metric="conversion_rate", test_type="t_test")
print(results)
```

## Visualization
The framework includes built-in visualizations for:
- Sample distributions.
- Test results with confidence intervals.
- Segment-wise differences.
