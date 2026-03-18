
This project analyzes cost report data from U.S. nursing homes (2015–2021) to help identify facilities suitable for investment. It was completed as part of the BANA 620: Predictive Analytics course at California State University, Northridge.

**`What I did`**

- Engineered financial and operational features (ROI, operating margin, debt ratios, bed occupancy, days in accounts receivable, etc.) from CMS cost report data (2015–2021).
- Trained and evaluated K-Nearest Neighbors, Logistic Regression, and Random Forest models; selected Random Forest with ~83.4% accuracy and 0.88 AUC.
- Analyzed pre- and post-COVID financial performance to highlight states and facilities with attractive investment potential (e.g., recommendations in Florida, Ohio, and California).

**`Objective`**

`To build a machine learning classification model that predicts whether a nursing home is a viable investment, using operational and financial metrics from CMS cost reports.`

**`Team Members`**

- `Venkat-Amit Kommineni`
- `Namrata Patil`
- `Krishnendu Nair`

**`Quickstart`**

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Open the main notebook:
    bash
    jupyter notebook notebooks/nh_investment_modeling.ipynb

**`Project Files`**
- notebooks/nh_investment_modeling.ipynb: Final notebook with modeling and results
- docs/NH_Investment_Report.html: Rendered version of the notebook
- NH_Data_Dictionary.pdf: Reference for data variables from CMS

**`Feature Engineering`**
- Return on Investment (ROI)
- Operating Margin
- Debt to Equity Ratio
- Current Ratio
- Bed Occupancy Rate
- Days in Accounts Receivable

**`Machine Learning Models`**
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest (best performance)

**`Key Insights`**
- Recommended investments in Florida, Ohio, and California
- 33% drop in net income post-COVID (2020–2021)
- Debt levels doubled in 2020 due to operational strain
- Random Forest yielded ~83.4% accuracy and 0.88 AUC

**`Tools & Technologies`**
- Python, Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- scikit-learn (for modeling and evaluation)
- PCA and cross-validation

**`Data Source`**
- CMS Nursing Home Cost Reports (2015–2021)
- Data Dictionary: NH_Data_Dictionary.pdf
  
**`License`**
- This project is released under the MIT License (see LICENSE).
