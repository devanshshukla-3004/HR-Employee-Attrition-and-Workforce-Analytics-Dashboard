"""
HR Employee Attrition and Workforce Analytics Dashboard
Built with Streamlit + Plotly on the IBM HR Analytics dataset.
Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
# Modern violet theme
COLORS = {"Yes": "#7C3AED", "No": "#C4B5FD"}
BAR_COLOR = "#7C3AED"
TEXT = "#2E1065"
SUBTEXT = "#7C6FA8"
GRID = "#EEEEF3"

st.markdown(
    """
    <style>
    .stApp { background: #F8F7FC; }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 6px; padding: 12px 14px; }
    [data-testid="stMetricLabel"] { color: #7C6FA8; }
    [data-testid="stMetricValue"] { color: #2E1065; font-weight: 600; }
    div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 6px; padding: 4px; }
    h1 { font-size: 1.7rem; font-weight: 600; color: #2E1065; padding-bottom: 0.5rem; }
    h2, h3 { color: #2E1065; font-weight: 600; }
    hr { border-color: #E5E5EA; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show(fig, container=st):
    """Apply the shared chart style and render."""
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(color=TEXT), title_font=dict(size=15, color=TEXT),
                      legend=dict(font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor=GRID, linecolor="#D5D5DD", zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor="#D5D5DD", zerolinecolor=GRID, automargin=True)
    container.plotly_chart(fig, width="stretch", theme=None)


# data
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Drop columns that carry no information (constant values / row IDs)
    df = df.drop(columns=[c for c in ["EmployeeCount", "StandardHours", "Over18"] if c in df.columns])
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 100],
                            labels=["18-25", "26-35", "36-45", "46-55", "56+"])
    df["IncomeBand"] = pd.cut(df["MonthlyIncome"], bins=[0, 3000, 6000, 10000, float("inf")],
                              labels=["< 3K", "3K-6K", "6K-10K", "10K+"])
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 100],
                              labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
    return df


def attrition_rate(data: pd.DataFrame, by: str) -> pd.DataFrame:
    """Attrition % and headcount for each category of `by`."""
    out = (data.groupby(by, observed=True)["AttritionFlag"]
           .agg(["mean", "sum", "count"]).reset_index())
    out.columns = [by, "Attrition Rate (%)", "Left", "Headcount"]
    out["Attrition Rate (%)"] = (out["Attrition Rate (%)"] * 100).round(1)
    return out


def rate_chart(data: pd.DataFrame, by: str, title: str, horizontal: bool = False, sort: bool = False):
    t = attrition_rate(data, by)
    if sort:
        t = t.sort_values("Attrition Rate (%)")
    x, y = ("Attrition Rate (%)", by) if horizontal else (by, "Attrition Rate (%)")
    fig = px.bar(t, x=x, y=y, text="Attrition Rate (%)", title=title,
                 hover_data=["Left", "Headcount"], orientation="h" if horizontal else "v")
    fig.update_traces(marker_color=BAR_COLOR, texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=380, margin=dict(t=50, b=10), yaxis_title=None if horizontal else "Attrition Rate (%)",
                      xaxis_title="Attrition Rate (%)" if horizontal else None)
    return fig


FEATURE_EXCLUDE = ["Attrition", "AttritionFlag", "AgeGroup", "IncomeBand", "TenureBand", "EmployeeNumber"]


def make_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000))


@st.cache_data
def train_model(df: pd.DataFrame):
    X = pd.get_dummies(df.drop(columns=FEATURE_EXCLUDE), drop_first=True)
    y = df["AttritionFlag"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = make_model().fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    # Coefficients on standardised features: sign = direction, size = strength
    coef = pd.Series(model[-1].coef_[0], index=X.columns)
    top = coef.reindex(coef.abs().sort_values(ascending=False).head(10).index).sort_values().reset_index()
    top.columns = ["Feature", "Effect"]
    top["Feature"] = top["Feature"].str.replace("_", ": ", n=1, regex=False)
    top["Direction"] = top["Effect"].apply(lambda v: "Raises attrition risk" if v > 0 else "Lowers attrition risk")
    metrics = {"accuracy": accuracy_score(y_te, pred), "recall": recall_score(y_te, pred),
               "precision": precision_score(y_te, pred, zero_division=0), "auc": roc_auc_score(y_te, prob)}
    cm = confusion_matrix(y_te, pred)
    # Out-of-fold probabilities: every employee is scored by a model that never saw their own record
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    risk = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1] * 100
    return top, metrics, cm, pd.Series(risk, index=df.index)


if not DATA_PATH.exists():
    st.error(f"Dataset not found. Place the Kaggle CSV at: {DATA_PATH}")
    st.stop()

df = load_data()

# sidebar
st.sidebar.header("Filters")
dept = st.sidebar.multiselect("Department", sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
roles = st.sidebar.multiselect("Job Role", sorted(df["JobRole"].unique()), default=sorted(df["JobRole"].unique()))
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
st.sidebar.caption("Source: IBM HR Analytics Employee Attrition & Performance (Kaggle)")

f = df[df["Department"].isin(dept) & df["Gender"].isin(gender) & df["JobRole"].isin(roles)
       & df["Age"].between(*age)]

# header
st.title("HR Employee Attrition and Workforce Analytics")

if f.empty:
    st.warning("No employees match the selected filters. Please widen your selection.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Employees", f"{len(f):,}")
k2.metric("Employees Left", f"{int(f['AttritionFlag'].sum()):,}")
k3.metric("Attrition Rate", f"{f['AttritionFlag'].mean() * 100:.1f}%")
k4.metric("Avg Monthly Income", f"${f['MonthlyIncome'].mean():,.0f}")
k5.metric("Avg Tenure", f"{f['YearsAtCompany'].mean():.1f} yrs")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Demographics", "Work & Compensation", "Left vs Stayed", "Attrition Drivers", "Risk List", "Data"])

# overview
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        pie = px.pie(f, names="Attrition", hole=0.55, color="Attrition", color_discrete_map=COLORS,
                     title="Attrition Split")
        pie.update_layout(height=380, margin=dict(t=50, b=10))
        show(pie, st)
    with c2:
        show(rate_chart(f, "Department", "Attrition Rate by Department"), st)
    show(rate_chart(f, "JobRole", "Attrition Rate by Job Role", horizontal=True, sort=True), st)

    st.subheader("Summary of findings")
    st.caption("Calculated from the currently selected filters.")
    dep_t = attrition_rate(f, "Department").sort_values("Attrition Rate (%)", ascending=False)
    role_t = attrition_rate(f, "JobRole").sort_values("Attrition Rate (%)", ascending=False)
    ot = attrition_rate(f, "OverTime").set_index("OverTime")["Attrition Rate (%)"]
    lines = [
        f"- {dep_t.iloc[0]['Department']} has the highest departmental attrition at "
        f"{dep_t.iloc[0]['Attrition Rate (%)']}%.",
        f"- {role_t.iloc[0]['JobRole']} is the highest-risk job role "
        f"({role_t.iloc[0]['Attrition Rate (%)']}%).",
    ]
    if {"Yes", "No"} <= set(ot.index) and ot["No"] > 0:
        lines.append(f"- Employees working overtime leave at {ot['Yes']}% vs {ot['No']}% for those who "
                     f"don't (about {ot['Yes'] / ot['No']:.1f}x higher).")
    left_inc, stay_inc = (f[f["Attrition"] == "Yes"]["MonthlyIncome"].mean(),
                          f[f["Attrition"] == "No"]["MonthlyIncome"].mean())
    if pd.notna(left_inc) and pd.notna(stay_inc):
        lines.append(f"- Employees who left earned ${left_inc:,.0f} per month on average vs "
                     f"${stay_inc:,.0f} for those who stayed.")
    st.markdown("\n".join(lines))

# demographics
with tab2:
    c1, c2 = st.columns(2)
    show(rate_chart(f, "AgeGroup", "Attrition Rate by Age Group"), c1)
    show(rate_chart(f, "Gender", "Attrition Rate by Gender"), c2)
    c3, c4 = st.columns(2)
    show(rate_chart(f, "MaritalStatus", "Attrition Rate by Marital Status"), c3)
    show(rate_chart(f, "EducationField", "Attrition Rate by Education Field", horizontal=True, sort=True), c4)
    hist = px.histogram(f, x="Age", color="Attrition", nbins=25, barmode="overlay", opacity=0.7,
                        color_discrete_map=COLORS, title="Age Distribution: Stayed vs Left")
    hist.update_layout(height=380, margin=dict(t=50, b=10))
    show(hist, st)

# ----------------------------------------------------- work & compensation ----
with tab3:
    c1, c2 = st.columns(2)
    show(rate_chart(f, "OverTime", "Attrition Rate: Overtime vs No Overtime"), c1)
    show(rate_chart(f, "IncomeBand", "Attrition Rate by Monthly Income Band"), c2)
    c3, c4 = st.columns(2)
    show(rate_chart(f, "JobSatisfaction", "Attrition Rate by Job Satisfaction (1 = Low, 4 = High)"), c3)
    show(rate_chart(f, "WorkLifeBalance", "Attrition Rate by Work-Life Balance (1 = Bad, 4 = Best)"), c4)
    c5, c6 = st.columns(2)
    show(rate_chart(f, "TenureBand", "Attrition Rate by Tenure"), c5)
    show(rate_chart(f, "YearsSinceLastPromotion", "Attrition Rate by Years Since Last Promotion"), c6)
    box = px.box(f, x="JobRole", y="MonthlyIncome", color="Attrition", color_discrete_map=COLORS,
                 title="Monthly Income by Job Role: Stayed vs Left")
    box.update_layout(height=450, margin=dict(t=50, b=10), xaxis_tickangle=-30)
    show(box, st)

# key drivers
with tab4:
    st.subheader("Employees who left compared with those who stayed")
    st.caption("Averages for the currently selected filters.")
    measures = {"Age": "Age", "Monthly income ($)": "MonthlyIncome", "Distance from home (km)": "DistanceFromHome",
                "Years at company": "YearsAtCompany", "Total working years": "TotalWorkingYears",
                "Years since last promotion": "YearsSinceLastPromotion", "Companies worked for": "NumCompaniesWorked",
                "Job satisfaction (1-4)": "JobSatisfaction", "Work-life balance (1-4)": "WorkLifeBalance"}
    g = f.groupby("Attrition")[list(measures.values())].mean().T
    for grp in ["No", "Yes"]:
        if grp not in g.columns:
            g[grp] = float("nan")
    cmp_tbl = pd.DataFrame({"Stayed": g["No"], "Left": g["Yes"]})
    cmp_tbl.index = list(measures.keys())
    ot = f.groupby("Attrition")["OverTime"].apply(lambda x: (x == "Yes").mean() * 100)
    cmp_tbl.loc["Works overtime (%)"] = [ot.get("No", float("nan")), ot.get("Yes", float("nan"))]
    cmp_tbl["Difference (%)"] = (cmp_tbl["Left"] - cmp_tbl["Stayed"]) / cmp_tbl["Stayed"] * 100
    st.dataframe(cmp_tbl.round(1), width="stretch")

    num_cols = ["Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
                "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
                "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
                "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
                "YearsWithCurrManager", "AttritionFlag"]
    corr = df[num_cols].corr()
    with_attr = corr["AttritionFlag"].drop("AttritionFlag").sort_values().reset_index()
    with_attr.columns = ["Factor", "Correlation"]
    bar = px.bar(with_attr, x="Correlation", y="Factor", orientation="h",
                 title="Correlation of each factor with attrition (full dataset)")
    bar.update_traces(marker_color=BAR_COLOR)
    bar.update_layout(height=520, margin=dict(t=50, b=10))
    show(bar, st)
    heat = px.imshow(corr, color_continuous_scale=[[0, "#5B8DEF"], [0.5, "#FFFFFF"], [1, "#7C3AED"]],
                     zmin=-1, zmax=1, title="Correlation matrix of numeric factors (full dataset)")
    heat.update_layout(height=650, margin=dict(t=50, b=10))
    show(heat, st)
    st.caption("Correlation shows how strongly two measures move together. It does not show cause and effect.")

with tab5:
    st.write("A class-balanced logistic regression model is trained on the full dataset (75/25 train-test split). "
             "Its coefficients show which factors push attrition risk up or down.")
    top, metrics, cm, risk = train_model(df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    m2.metric("Recall (left)", f"{metrics['recall'] * 100:.1f}%")
    m3.metric("Precision (left)", f"{metrics['precision'] * 100:.1f}%")
    m4.metric("ROC AUC", f"{metrics['auc']:.2f}")
    c1, c2 = st.columns(2)
    fig = px.bar(top, x="Effect", y="Feature", orientation="h", color="Direction",
                 color_discrete_map={"Raises attrition risk": "#7C3AED", "Lowers attrition risk": "#C4B5FD"},
                 title="Top 10 factors by effect on attrition risk")
    fig.update_layout(height=440, margin=dict(t=50, b=10), legend_title_text="", legend=dict(orientation="h", y=-0.15),
                      xaxis_title="Standardised coefficient")
    show(fig, c1)
    cm_fig = px.imshow(cm, x=["Predicted: stay", "Predicted: leave"], y=["Actual: stayed", "Actual: left"],
                       text_auto=True, color_continuous_scale=["#FFFFFF", "#7C3AED"], title="Confusion matrix (test set)")
    cm_fig.update_layout(height=440, margin=dict(t=50, b=10), coloraxis_showscale=False)
    show(cm_fig, c2)
    st.caption("Only about 16% of employees left, so accuracy alone can look good even for a weak model. "
               "Recall shows how many of the employees who actually left the model caught; precision shows how many "
               "of its 'leave' predictions were correct. Coefficients show association, not causation.")

with tab6:
    st.subheader("Current employees ranked by estimated attrition risk")
    st.caption("Scores come from a cross-validated logistic regression model, so each employee is scored by a model that did not "
               "see their own record. Use them to prioritise conversations, not to make decisions about individuals.")
    _, _, _, risk = train_model(df)
    current = f[f["Attrition"] == "No"].copy()
    if current.empty:
        st.info("No current employees match the selected filters.")
    else:
        current["Risk Score (%)"] = risk.loc[current.index].round(1)
        n_show = st.slider("Number of employees to show", 10, 100, 25, 5)
        show_cols = ["EmployeeNumber", "Department", "JobRole", "Age", "MonthlyIncome", "OverTime",
                     "YearsAtCompany", "JobSatisfaction", "Risk Score (%)"]
        top = current.sort_values("Risk Score (%)", ascending=False).head(n_show)[show_cols]
        st.dataframe(top.reset_index(drop=True), width="stretch", hide_index=True)
        st.download_button("Download risk list (CSV)", top.to_csv(index=False).encode("utf-8"),
                           file_name="attrition_risk_list.csv", mime="text/csv")

with tab7:
    with st.expander("About the data"):
        st.markdown(
            """
            The data is the IBM HR Analytics Employee Attrition and Performance dataset from Kaggle: 1,470 employee
            records with 35 columns. It is a fictional dataset created by IBM data scientists for analytics practice.

            Columns removed before analysis: `EmployeeCount`, `StandardHours` and `Over18` (same value for every
            row). `EmployeeNumber` is kept only as an identifier and is not used by the model.

            Rating scales used in the dataset:

            | Column | 1 | 2 | 3 | 4 | 5 |
            |---|---|---|---|---|---|
            | Education | Below college | College | Bachelor | Master | Doctor |
            | EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, JobInvolvement | Low | Medium | High | Very high | - |
            | WorkLifeBalance | Bad | Good | Better | Best | - |
            | PerformanceRating | Low | Good | Excellent | Outstanding | - |

            Derived columns: `AgeGroup`, `IncomeBand`, `TenureBand` and `AttritionFlag` (1 = left, 0 = stayed).
            Monthly income is shown in dollars as given in the dataset.
            """
        )
    st.dataframe(f.drop(columns=["AttritionFlag"]), width="stretch", height=450)
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_hr_data.csv", mime="text/csv")
