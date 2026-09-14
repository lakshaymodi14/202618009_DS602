# ============================================================
# DS602 Lab-4
# Applied Statistical Modeling & Interactive Web Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Restaurant Tips Dashboard",
    page_icon="🍽️",
    layout="wide"
)


# ============================================================
# Constants
# ============================================================

PREDICTORS = [
    "total_bill",
    "size",
    "sex",
    "smoker",
    "day",
    "time"
]

CATEGORICAL = [
    "sex",
    "smoker",
    "day",
    "time"
]

CATEGORIES = {
    "sex": ["Female", "Male"],
    "smoker": ["No", "Yes"],
    "day": ["Thur", "Fri", "Sat", "Sun"],
    "time": ["Dinner", "Lunch"]
}


# ============================================================
# Load and Prepare Data
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("tips_clean.csv")

    # Restore the same categorical ordering used in the notebook
    data["sex"] = pd.Categorical(
        data["sex"],
        categories=CATEGORIES["sex"]
    )

    data["smoker"] = pd.Categorical(
        data["smoker"],
        categories=CATEGORIES["smoker"]
    )

    data["day"] = pd.Categorical(
        data["day"],
        categories=CATEGORIES["day"],
        ordered=True
    )

    data["time"] = pd.Categorical(
        data["time"],
        categories=CATEGORIES["time"]
    )

    return data


df = load_data()


# ============================================================
# Build Regression Model
# ============================================================

@st.cache_resource
def build_model(data):

    # Same 80/20 split used in the notebook
    train_df, test_df = train_test_split(
        data,
        test_size=0.20,
        random_state=42
    )

    # Training design matrix
    X_train = pd.get_dummies(
        train_df[PREDICTORS],
        columns=CATEGORICAL,
        drop_first=True,
        dtype=float
    )

    X_train = sm.add_constant(
        X_train,
        has_constant="add"
    )

    y_train = train_df["tip"]

    # OLS model
    model = sm.OLS(
        y_train,
        X_train
    ).fit()

    # Test design matrix
    X_test = pd.get_dummies(
        test_df[PREDICTORS],
        columns=CATEGORICAL,
        drop_first=True,
        dtype=float
    )

    # Make test columns exactly match training columns
    X_test = X_test.reindex(
        columns=X_train.columns.drop("const"),
        fill_value=0
    )

    X_test = sm.add_constant(
        X_test,
        has_constant="add"
    )

    y_test = test_df["tip"]

    # Test predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    test_r2 = r2_score(
        y_test,
        y_pred
    )

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": test_r2
    }

    return (
        model,
        X_train.columns,
        metrics,
        train_df,
        test_df
    )


model, model_columns, test_metrics, train_df, test_df = build_model(df)


# ============================================================
# Helper Function for Prediction Encoding
# ============================================================

def encode_prediction(input_df):

    input_df = input_df.copy()

    # Apply exactly the same categories as training data
    input_df["sex"] = pd.Categorical(
        input_df["sex"],
        categories=CATEGORIES["sex"]
    )

    input_df["smoker"] = pd.Categorical(
        input_df["smoker"],
        categories=CATEGORIES["smoker"]
    )

    input_df["day"] = pd.Categorical(
        input_df["day"],
        categories=CATEGORIES["day"],
        ordered=True
    )

    input_df["time"] = pd.Categorical(
        input_df["time"],
        categories=CATEGORIES["time"]
    )

    X = pd.get_dummies(
        input_df[PREDICTORS],
        columns=CATEGORICAL,
        drop_first=True,
        dtype=float
    )

    # Force exact same columns as training model
    X = X.reindex(
        columns=model_columns.drop("const"),
        fill_value=0
    )

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    return X


# ============================================================
# Title
# ============================================================

st.title(
    "🍽️ Restaurant Tips — Statistical Modeling Dashboard"
)

st.write(
    "Interactive exploration, hypothesis testing, "
    "and regression-based tip prediction."
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Dashboard Controls")

st.sidebar.write(
    f"Dataset: **{len(df)} observations**"
)

st.sidebar.write(
    f"Variables: **{df.shape[1]} columns**"
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "🔮 Live Prediction & Diagnostics"
])


# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header("Data Exploration")

    # --------------------------------------------------------
    # Sidebar Filters
    # --------------------------------------------------------

    st.sidebar.subheader("Data Filters")

    selected_sex = st.sidebar.multiselect(
        "Sex",
        options=CATEGORIES["sex"],
        default=CATEGORIES["sex"]
    )

    selected_smoker = st.sidebar.multiselect(
        "Smoker",
        options=CATEGORIES["smoker"],
        default=CATEGORIES["smoker"]
    )

    selected_day = st.sidebar.multiselect(
        "Day",
        options=CATEGORIES["day"],
        default=CATEGORIES["day"]
    )

    selected_time = st.sidebar.multiselect(
        "Time",
        options=CATEGORIES["time"],
        default=CATEGORIES["time"]
    )

    # --------------------------------------------------------
    # Apply Filters
    # --------------------------------------------------------

    filtered_df = df[
        df["sex"].isin(selected_sex)
        & df["smoker"].isin(selected_smoker)
        & df["day"].isin(selected_day)
        & df["time"].isin(selected_time)
    ]

    # --------------------------------------------------------
    # Empty Filter Protection
    # --------------------------------------------------------

    if len(filtered_df) == 0:

        st.warning(
            "No observations match the selected filters."
        )

    else:

        # ----------------------------------------------------
        # Summary Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Observations",
            len(filtered_df)
        )

        col2.metric(
            "Average Bill",
            f"${filtered_df['total_bill'].mean():.2f}"
        )

        col3.metric(
            "Average Tip",
            f"${filtered_df['tip'].mean():.2f}"
        )

        col4.metric(
            "Average Tip %",
            f"{filtered_df['tip_pct'].mean():.2f}%"
        )

        # ----------------------------------------------------
        # Filtered Data
        # ----------------------------------------------------

        st.subheader("Filtered Data")

        st.dataframe(
            filtered_df,
            width="stretch"
        )

        # ----------------------------------------------------
        # Interactive Plots
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Bill vs Tip")

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            sns.scatterplot(
                data=filtered_df,
                x="total_bill",
                y="tip",
                hue="smoker",
                style="time",
                ax=ax
            )

            ax.set_title(
                "Total Bill vs Tip"
            )

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            st.subheader("Tip Distribution")

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            sns.histplot(
                filtered_df["tip"],
                kde=True,
                ax=ax
            )

            ax.set_title(
                "Distribution of Tips"
            )

            ax.set_xlabel("Tip")

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ============================================================

with tab2:

    st.header("Hypothesis Testing Lab")

    st.write(
        "Select a categorical grouping variable and a numerical "
        "variable to perform an appropriate statistical test."
    )

    categorical_options = [
        "smoker",
        "sex",
        "day",
        "time"
    ]

    numerical_options = [
        "total_bill",
        "tip",
        "size",
        "tip_pct"
    ]

    group_col = st.selectbox(
        "Categorical Grouping Variable",
        categorical_options
    )

    numeric_col = st.selectbox(
        "Numerical Variable",
        numerical_options
    )

    groups = list(
        df[group_col].dropna().unique()
    )

    # ========================================================
    # TWO GROUPS
    # ========================================================

    if len(groups) == 2:

        st.subheader(
            "Two-Group Comparison"
        )

        group1 = df[
            df[group_col] == groups[0]
        ][numeric_col].dropna()

        group2 = df[
            df[group_col] == groups[1]
        ][numeric_col].dropna()

        # ----------------------------------------------------
        # Group Summary
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**{groups[0]}**"
            )

            st.write(
                f"n = {len(group1)}"
            )

            st.write(
                f"Mean = {group1.mean():.3f}"
            )

            st.write(
                f"Std Dev = {group1.std():.3f}"
            )

        with col2:

            st.write(
                f"**{groups[1]}**"
            )

            st.write(
                f"n = {len(group2)}"
            )

            st.write(
                f"Mean = {group2.mean():.3f}"
            )

            st.write(
                f"Std Dev = {group2.std():.3f}"
            )

        # ----------------------------------------------------
        # Shapiro-Wilk Tests
        # ----------------------------------------------------

        shapiro1 = stats.shapiro(group1)
        shapiro2 = stats.shapiro(group2)

        st.subheader(
            "Shapiro-Wilk Normality Tests"
        )

        shapiro_table = pd.DataFrame({
            "Group": [
                str(groups[0]),
                str(groups[1])
            ],
            "Statistic": [
                shapiro1.statistic,
                shapiro2.statistic
            ],
            "p-value": [
                shapiro1.pvalue,
                shapiro2.pvalue
            ]
        })

        st.dataframe(
            shapiro_table.round(4),
            width="stretch"
        )

        normal = (
            shapiro1.pvalue >= 0.05
            and
            shapiro2.pvalue >= 0.05
        )

        # ----------------------------------------------------
        # Parametric Case
        # ----------------------------------------------------

        if normal:

            st.info(
                "Both groups pass the Shapiro-Wilk normality "
                "check at α = 0.05. Proceeding with Levene's test."
            )

            levene = stats.levene(
                group1,
                group2
            )

            st.write(
                f"**Levene's test statistic:** "
                f"{levene.statistic:.4f}"
            )

            st.write(
                f"**Levene's test p-value:** "
                f"{levene.pvalue:.4f}"
            )

            equal_var = (
                levene.pvalue >= 0.05
            )

            statistic, p_value = stats.ttest_ind(
                group1,
                group2,
                equal_var=equal_var
            )

            test_name = (
                "Independent Two-Sample t-test"
            )

            st.write(
                f"**Test selected:** {test_name}"
            )

            st.write(
                "**H₀:** The population means of the two "
                "groups are equal."
            )

            st.write(
                "**H₁:** The population means of the two "
                "groups are different."
            )

        # ----------------------------------------------------
        # Non-Parametric Case
        # ----------------------------------------------------

        else:

            st.info(
                "At least one group fails the Shapiro-Wilk "
                "normality check at α = 0.05. "
                "Using the Mann-Whitney U test."
            )

            statistic, p_value = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            test_name = (
                "Mann-Whitney U Test"
            )

            st.write(
                f"**Test selected:** {test_name}"
            )

            st.write(
                "**H₀:** The two groups have the same "
                "distribution/location."
            )

            st.write(
                "**H₁:** The two groups have different "
                "distributions/locations."
            )

        # ----------------------------------------------------
        # Test Result
        # ----------------------------------------------------

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Test Statistic",
            f"{statistic:.4f}"
        )

        col2.metric(
            "p-value",
            f"{p_value:.4f}"
        )

        if p_value < 0.05:

            st.error(
                "Reject H₀: There is statistically significant "
                "evidence of a difference between the groups."
            )

        else:

            st.success(
                "Fail to reject H₀: There is insufficient "
                "evidence of a statistically significant "
                "difference between the groups."
            )

    # ========================================================
    # MORE THAN TWO GROUPS
    # ========================================================

    else:

        st.subheader(
            "Multiple-Group Comparison"
        )

        grouped_data = [
            df[
                df[group_col] == group
            ][numeric_col].dropna()
            for group in groups
        ]

        statistic, p_value = stats.f_oneway(
            *grouped_data
        )

        st.write(
            "**Test selected:** One-Way ANOVA"
        )

        st.write(
            "**H₀:** All group population means are equal."
        )

        st.write(
            "**H₁:** At least one group population mean "
            "is different."
        )

        # ----------------------------------------------------
        # Group Summary Table
        # ----------------------------------------------------

        anova_summary = pd.DataFrame({
            "Group": [
                str(group)
                for group in groups
            ],
            "n": [
                len(values)
                for values in grouped_data
            ],
            "Mean": [
                values.mean()
                for values in grouped_data
            ],
            "Std Dev": [
                values.std()
                for values in grouped_data
            ]
        })

        st.dataframe(
            anova_summary.round(3),
            width="stretch"
        )

        # ----------------------------------------------------
        # ANOVA Result
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        col1.metric(
            "F-statistic",
            f"{statistic:.4f}"
        )

        col2.metric(
            "p-value",
            f"{p_value:.4f}"
        )

        if p_value < 0.05:

            st.error(
                "Reject H₀: At least one group mean differs "
                "significantly from the others."
            )

        else:

            st.success(
                "Fail to reject H₀: There is insufficient "
                "evidence that the group means differ."
            )


# ============================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header(
        "Live Tip Prediction & Diagnostics"
    )

    st.write(
        "Enter customer and bill information to obtain "
        "a real-time regression prediction."
    )

    # --------------------------------------------------------
    # User Inputs
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        total_bill_input = st.number_input(
            "Total Bill",
            min_value=0.01,
            value=20.00,
            step=1.00
        )

        size_input = st.number_input(
            "Party Size",
            min_value=1,
            value=2,
            step=1
        )

    with col2:

        sex_input = st.selectbox(
            "Sex",
            CATEGORIES["sex"]
        )

        smoker_input = st.selectbox(
            "Smoker",
            CATEGORIES["smoker"]
        )

    with col3:

        day_input = st.selectbox(
            "Day",
            CATEGORIES["day"]
        )

        time_input = st.selectbox(
            "Time",
            CATEGORIES["time"]
        )

    # --------------------------------------------------------
    # Create Prediction Row
    # --------------------------------------------------------

    input_df = pd.DataFrame({
        "total_bill": [total_bill_input],
        "size": [size_input],
        "sex": [sex_input],
        "smoker": [smoker_input],
        "day": [day_input],
        "time": [time_input]
    })

    input_encoded = encode_prediction(
        input_df
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_result = model.get_prediction(
        input_encoded
    )

    prediction_summary = prediction_result.summary_frame(
        alpha=0.05
    )

    predicted_tip = (
        prediction_summary["mean"].iloc[0]
    )

    mean_lower = (
        prediction_summary["mean_ci_lower"].iloc[0]
    )

    mean_upper = (
        prediction_summary["mean_ci_upper"].iloc[0]
    )

    pred_lower = (
        prediction_summary["obs_ci_lower"].iloc[0]
    )

    pred_upper = (
        prediction_summary["obs_ci_upper"].iloc[0]
    )

    # --------------------------------------------------------
    # Prediction Display
    # --------------------------------------------------------

    st.subheader("Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Tip",
        f"${predicted_tip:.2f}"
    )

    col2.metric(
        "95% Confidence Interval",
        f"${mean_lower:.2f} – ${mean_upper:.2f}"
    )

    col3.metric(
        "95% Prediction Interval",
        f"${pred_lower:.2f} – ${pred_upper:.2f}"
    )

    st.caption(
        "The confidence interval estimates the mean tip for "
        "customers with these characteristics. The prediction "
        "interval gives a wider range for an individual future tip."
    )

    # --------------------------------------------------------
    # Regression Diagnostics
    # --------------------------------------------------------

    st.subheader(
        "Regression Diagnostics"
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.scatterplot(
            x=model.fittedvalues,
            y=model.resid,
            ax=ax
        )

        ax.axhline(
            0,
            linestyle="--"
        )

        ax.set_xlabel(
            "Fitted Values"
        )

        ax.set_ylabel(
            "Residuals"
        )

        ax.set_title(
            "Residuals vs Fitted Values"
        )

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        fig = plt.figure(
            figsize=(7, 5)
        )

        sm.qqplot(
            model.resid,
            line="45",
            fit=True
        )

        plt.title(
            "Q-Q Plot of Regression Residuals"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Model Performance
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test MAE",
        f"{test_metrics['MAE']:.4f}"
    )

    col2.metric(
        "Test RMSE",
        f"{test_metrics['RMSE']:.4f}"
    )

    col3.metric(
        "Test R²",
        f"{test_metrics['R²']:.4f}"
    )

    col4.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.4f}"
    )

    # --------------------------------------------------------
    # Regression Coefficients
    # --------------------------------------------------------

    st.subheader(
        "Regression Coefficients"
    )

    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "p-value": model.pvalues,
        "Lower 95%": model.conf_int()[0],
        "Upper 95%": model.conf_int()[1]
    })

    st.dataframe(
        coefficient_table.round(4),
        width="stretch"
    )