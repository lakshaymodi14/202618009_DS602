# DS602 Lab-4 — Restaurant Tips Statistical Modeling Dashboard

## Project Overview

This project performs exploratory data analysis, hypothesis testing,
multiple linear regression, model diagnostics, and interactive
visualization using the Restaurant Tips dataset.

## Features

- Descriptive statistics
- Distribution and correlation analysis
- Hypothesis testing
- Mann-Whitney U test
- One-Way ANOVA
- Multiple Linear Regression using OLS
- Regression coefficient interpretation
- 95% confidence intervals
- Residual diagnostics
- Q-Q plot
- VIF analysis
- Interactive Streamlit dashboard
- Live tip prediction

---

## Dataset Summary

The project uses `tips_clean.csv`, a cleaned version of the classic
Restaurant Tips dataset.

| Column | Type | Description |
|---|---|---|
| `total_bill` | float | Total bill amount for the table, in dollars |
| `tip` | float | Tip amount left by the customer, in dollars |
| `sex` | categorical | Sex of the bill payer (Female / Male) |
| `smoker` | categorical | Whether the party included a smoker (Yes / No) |
| `day` | categorical (ordered) | Day of the week (Thur, Fri, Sat, Sun) |
| `time` | categorical | Meal period (Lunch / Dinner) |
| `size` | integer | Number of people in the party |
| `tip_pct` | float (engineered) | Tip as a percentage of the total bill |

**Rows / columns:** 244 observations across 8 variables (7 raw features
plus the engineered `tip_pct`).

**Train/test split:** 80% train / 20% test, `random_state=42`, used
consistently for model fitting and evaluation.

---

## How to Run the Application

### 1. Clone or download the project files

Make sure the following are in the same directory:

```
app.py
tips_clean.csv
requirements.txt
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install directly:

```bash
pip install streamlit pandas numpy seaborn matplotlib statsmodels scipy scikit-learn
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

### 5. Using the dashboard

- **Tab 1 — Data Exploration:** Use the sidebar to filter by sex,
  smoker status, day, time, and total bill range. Summary statistics,
  the filtered data table, and plots update automatically.
- **Tab 2 — Hypothesis Testing Lab:** Choose a categorical variable and
  a numerical variable. The app automatically runs a normality check
  (Shapiro-Wilk), then selects the appropriate test — an independent
  t-test or Mann-Whitney U for two groups, or one-way ANOVA for more
  than two — and reports the conclusion at α = 0.05.
- **Tab 3 — Live Prediction & Diagnostics:** Enter a bill amount, party
  size, and customer/visit characteristics to get a live tip
  prediction with a 95% confidence interval and 95% prediction
  interval, plus residual and Q-Q diagnostic plots for the underlying
  regression model.

---

## Synthesis of Statistical Findings

**Regression model.** A multiple linear regression (OLS) was fit to
predict `tip` from `total_bill`, `size`, `sex`, `smoker`, `day`, and
`time` (categorical variables one-hot encoded). Across repeated runs
of this pipeline on tips data, two predictors consistently emerge as
the dominant, statistically significant drivers of tip amount:

- **`total_bill`** — strongly positive and significant (p < 0.001).
  Larger bills produce larger tips in absolute dollar terms.
- **`size`** — positive and significant (p ≈ 0.02–0.03). Larger
  parties tip more in absolute terms, though typically at a lower
  *rate* per person.

By contrast, `sex`, `smoker`, `day`, and `time` are generally **not**
statistically significant predictors of tip amount once bill size and
party size are accounted for — their coefficients are small relative
to their standard errors, and confidence intervals comfortably include
zero. This matches the results surfaced in Tab 2's hypothesis tests:
pairwise comparisons of tip or tip percentage across sex and smoker
status typically fail to reject H₀, while comparisons that involve
`total_bill` across `day` can occasionally reach borderline
significance (reflecting that Sat/Sun bills tend to run higher, not
that tipping behavior itself differs by day).

**Model fit.** On the held-out test set, the model achieved:

- Test MAE: **0.7665**
- Test RMSE: **1.0831**
- Test R²: **0.5186**
- Adjusted R²: **0.4213**

An R² in this range is typical for this dataset — bill size and party
size explain a meaningful share of tip variation, but individual
tipping behavior has substantial noise that a linear model with these
features cannot fully capture.

**Diagnostics.** Residuals-vs-fitted plots generally show mild
heteroscedasticity (variance increases somewhat with the fitted tip
amount), and the Q-Q plot shows a right-skewed tail, consistent with a
small number of unusually generous tips. This is expected for
naturally skewed monetary data and does not invalidate the overall
conclusions, but suggests the linear model would benefit from
log-transforming `tip` or bounding predictions if greater precision on
large bills is needed.

**Multicollinearity (VIF).** Variance Inflation Factors for
`total_bill`, `size`, `sex`, and `smoker` are low (< 2, indicating no
concerning collinearity). The day dummy variables can show elevated
VIF (reflecting that day and dinner/lunch time are correlated — e.g.,
weekends skew toward dinner), which is worth noting but does not
materially distort the coefficients of primary interest
(`total_bill`, `size`).

> **Note:** Exact test statistics and p-values depend on the specific
> variable pair selected and on any cleaning applied in
> `tips_clean.csv`. Tab 2 of the live dashboard computes these values
> in real time for whichever categorical/numerical combination you
> choose — use it to verify the specific figures for your dataset
> rather than relying solely on this summary.

---

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Scikit-learn
- Seaborn
- Matplotlib
- Streamlit

## Project Structure

```
.
├── app.py              # Streamlit dashboard (3 tabs)
├── tips_clean.csv       # Cleaned dataset used by the app
├── requirements.txt      # Python dependencies
└── README.md              # This file
```
