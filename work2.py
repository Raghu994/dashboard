# dashboard_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Sleep Health Dashboard", layout="wide")

# -------------------------
# Embedded CSS for styling
# -------------------------
st.markdown(
    """
    <style>
    /* App background and fonts */
    .stApp {
        background-color: #071023;
        color: #E6EEF6;
    }
    /* KPI card style */
    .kpi-card {
        background: linear-gradient(135deg, rgba(17,24,39,0.85), rgba(8,16,27,0.95));
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        text-align: center;
        margin: 6px 4px;
    }
    .kpi-title { color: #9AA9BF; font-size: 13px; margin:0 0 6px 0; }
    .kpi-value { color: #FFF; font-size: 26px; margin:0; font-weight:700; }
    .kpi-sub { color:#9AA9BF; font-size:12px; margin-top:6px; }
    /* Compact card for predictor */
    .panel-card {
        background: linear-gradient(180deg, rgba(7,10,20,0.9), rgba(9,15,30,0.95));
        padding:12px;
        border-radius:10px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
    }
    /* smaller margins for wider layout */
    .block-container { padding-top: 10px; padding-left: 16px; padding-right: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper functions
# -------------------------
def load_data(path="Sleep_health_and_lifestyle_dataset.csv"):
    return pd.read_csv(path)

def try_parse_bp(bp):
    try:
        if pd.isna(bp): return np.nan, np.nan
        s = str(bp)
        if "/" in s:
            a, b = s.split("/")
            return float(a), float(b)
        else:
            return float(s), np.nan
    except:
        return np.nan, np.nan

def norm_series(s):
    if s is None or s.size == 0:
        return s
    mn = np.nanmin(s)
    mx = np.nanmax(s)
    if np.isnan(mn) or np.isnan(mx) or mn == mx:
        return (s - mn) if not np.all(np.isnan(s)) else s
    return (s - mn) / (mx - mn)

def age_bucket(age):
    try:
        a = float(age)
    except:
        return "Unknown"
    if a < 18: return "<18"
    if a < 30: return "18-29"
    if a < 45: return "30-44"
    if a < 60: return "45-59"
    return "60+"

# -------------------------
# Load + validate dataset
# -------------------------
@st.cache_data
def _load(path="Sleep_health_and_lifestyle_dataset.csv"):
    return load_data(path)

try:
    df = _load()
except FileNotFoundError:
    st.error("Dataset file not found. Put 'Sleep_health_and_lifestyle_dataset.csv' next to this script.")
    st.stop()
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Ensure Person ID exists
if 'Person ID' not in df.columns:
    df['Person ID'] = df.index.astype(str)

# Coerce numeric columns safely
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'Stress Level',
                'Heart Rate', 'Daily Steps']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Parse Blood Pressure into systolic/diastolic
if 'Blood Pressure' in df.columns:
    bp = df['Blood Pressure'].apply(try_parse_bp)
    df['Systolic'] = [x[0] for x in bp]
    df['Diastolic'] = [x[1] for x in bp]
else:
    df['Systolic'] = np.nan
    df['Diastolic'] = np.nan

# Age bucket
df['Age Bucket'] = df['Age'].apply(age_bucket) if 'Age' in df.columns else 'Unknown'

# Add placeholders for Sleep Efficiency/Awakenings (may be added if merging external datasets)
if 'Sleep Efficiency' not in df.columns:
    df['Sleep Efficiency'] = np.nan
if 'Awakenings' not in df.columns:
    df['Awakenings'] = np.nan

# Composite SQI (Sleep Quality Index): combine normalized Quality & Duration
df['Quality_norm'] = norm_series(df['Quality of Sleep']) if 'Quality of Sleep' in df.columns else np.nan
df['Duration_norm'] = norm_series(df['Sleep Duration']) if 'Sleep Duration' in df.columns else np.nan
# weights: Quality 0.6, Duration 0.4 (tweakable)
df['SQI'] = 0.6 * df['Quality_norm'].fillna(0) + 0.4 * df['Duration_norm'].fillna(0)

# Active Flag (>=30 minutes activity considered active)
df['Active Flag'] = (df['Physical Activity Level'] >= 30).astype(int) if 'Physical Activity Level' in df.columns else 0

# -------------------------
# Sidebar filters (compact)
# -------------------------
st.sidebar.header("Filters")
gender_options = ['All'] + sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else ['All']
occupation_options = ['All'] + sorted(df['Occupation'].dropna().unique().tolist()) if 'Occupation' in df.columns else ['All']
agebucket_options = ['All'] + sorted(df['Age Bucket'].dropna().unique().tolist()) if 'Age Bucket' in df.columns else ['All']
bmi_options = ['All'] + sorted(df['BMI Category'].dropna().unique().tolist()) if 'BMI Category' in df.columns else ['All']

selected_gender = st.sidebar.selectbox("Gender", options=gender_options, index=0)
selected_occupation = st.sidebar.selectbox("Occupation", options=occupation_options, index=0)
selected_agebucket = st.sidebar.selectbox("Age Bucket", options=agebucket_options, index=0)
selected_bmi = st.sidebar.selectbox("BMI Category", options=bmi_options, index=0)

# # small upload area if user wants to bring another CSV (not merging in this minimal flow)
# st.sidebar.markdown("---")
# st.sidebar.markdown("**Upload (optional):**")
# uploaded_file = st.sidebar.file_uploader("Upload CSV to replace dataset", type=["csv"])
# if uploaded_file is not None:
#     try:
#         uploaded_df = pd.read_csv(uploaded_file)
#         uploaded_df.columns = [c.strip() for c in uploaded_df.columns]
#         st.sidebar.success("Uploaded — preview shown below")
#         st.sidebar.write(uploaded_df.head(3))
#         # use uploaded as df (non-persistent unless rerun)
#         df = uploaded_df.copy()
#         # re-run the preprocessing steps quickly
#         for c in numeric_cols:
#             if c in df.columns:
#                 df[c] = pd.to_numeric(df[c], errors='coerce')
#         if 'Blood Pressure' in df.columns:
#             bp = df['Blood Pressure'].apply(try_parse_bp)
#             df['Systolic'] = [x[0] for x in bp]
#             df['Diastolic'] = [x[1] for x in bp]
#         else:
#             df['Systolic'] = np.nan; df['Diastolic'] = np.nan
#         df['Age Bucket'] = df['Age'].apply(age_bucket) if 'Age' in df.columns else 'Unknown'
#         if 'Sleep Efficiency' not in df.columns:
#             df['Sleep Efficiency'] = np.nan
#         if 'Awakenings' not in df.columns:
#             df['Awakenings'] = np.nan
#         df['Quality_norm'] = norm_series(df['Quality of Sleep']) if 'Quality of Sleep' in df.columns else np.nan
#         df['Duration_norm'] = norm_series(df['Sleep Duration']) if 'Sleep Duration' in df.columns else np.nan
#         df['SQI'] = 0.6 * df['Quality_norm'].fillna(0) + 0.4 * df['Duration_norm'].fillna(0)
#         df['Active Flag'] = (df['Physical Activity Level'] >= 30).astype(int) if 'Physical Activity Level' in df.columns else 0
#     except Exception as e:
#         st.sidebar.error(f"Upload error: {e}")

# -------------------------
# Filter dataframe according to sidebar
# -------------------------
df_filtered = df.copy()
if selected_gender != 'All' and 'Gender' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
if selected_occupation != 'All' and 'Occupation' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Occupation'] == selected_occupation]
if selected_agebucket != 'All' and 'Age Bucket' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Age Bucket'] == selected_agebucket]
if selected_bmi != 'All' and 'BMI Category' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['BMI Category'] == selected_bmi]

# -------------------------
# Multi-KPI calculations
# -------------------------
n_participants = len(df_filtered)
avg_sleep = df_filtered['Sleep Duration'].mean() if 'Sleep Duration' in df_filtered.columns else np.nan
avg_quality = df_filtered['Quality of Sleep'].mean() if 'Quality of Sleep' in df_filtered.columns else np.nan
avg_sqi = df_filtered['SQI'].mean() if 'SQI' in df_filtered.columns else np.nan
sleep_std = None
if 'Person ID' in df_filtered.columns and df_filtered.duplicated(subset=['Person ID']).any():
    # if there are multiple rows per Person ID, compute avg per-person variability
    per_person_std = df_filtered.groupby('Person ID')['Sleep Duration'].std().dropna()
    sleep_std = per_person_std.mean() if not per_person_std.empty else np.nan
else:
    # dataset likely cross-sectional - show dataset-level variability
    sleep_std = df_filtered['Sleep Duration'].std() if 'Sleep Duration' in df_filtered.columns else np.nan

avg_hr = df_filtered['Heart Rate'].mean() if 'Heart Rate' in df_filtered.columns else np.nan
avg_steps = df_filtered['Daily Steps'].mean() if 'Daily Steps' in df_filtered.columns else np.nan
pct_active = (df_filtered['Active Flag'].mean() * 100) if 'Active Flag' in df_filtered.columns else np.nan

# Sleep disorders breakdown
if 'Sleep Disorder' in df_filtered.columns:
    disorder_counts = df_filtered['Sleep Disorder'].value_counts(dropna=False)
    pct_any_disorder = (df_filtered['Sleep Disorder'] != 'None').sum() / n_participants * 100 if n_participants>0 else 0
    pct_insomnia = (df_filtered['Sleep Disorder'] == 'Insomnia').sum() / n_participants * 100 if n_participants>0 else 0
    pct_apnea = (df_filtered['Sleep Disorder'] == 'Sleep Apnea').sum() / n_participants * 100 if n_participants>0 else 0
else:
    pct_any_disorder = pct_insomnia = pct_apnea = np.nan

# High blood pressure %
if 'Systolic' in df_filtered.columns and 'Diastolic' in df_filtered.columns:
    pct_high_bp = ((df_filtered['Systolic'] >= 130) | (df_filtered['Diastolic'] >= 80)).mean() * 100
else:
    pct_high_bp = np.nan

# -------------------------
# KPI row (compact, single line)
# -------------------------
st.markdown("<div style='display:flex; gap:8px; align-items:stretch;'>", unsafe_allow_html=True)
kpi_cols = st.columns([1,1,1,1,1,1,1])
# create list of (title, value, subtitle)
kpis = [
    ("Participants", f"{n_participants}", "Total rows after filters"),
    ("Avg Sleep (hrs)", f"{avg_sleep:.2f}" if not np.isnan(avg_sleep) else "N/A", "Mean hours"),
    ("Avg Quality (1-10)", f"{avg_quality:.2f}" if not np.isnan(avg_quality) else "N/A", ""),
    ("Avg SQI", f"{avg_sqi:.2f}" if not np.isnan(avg_sqi) else "N/A", "Composite score 0-1"),
    ("Sleep Variability (SD)", f"{sleep_std:.2f}" if not np.isnan(sleep_std) else "N/A", ""),
    ("Avg Resting HR", f"{avg_hr:.0f} bpm" if not np.isnan(avg_hr) else "N/A", ""),
    ("% Any Disorder", f"{pct_any_disorder:.1f}%" if not np.isnan(pct_any_disorder) else "N/A", "")
]

for col, (title, value, sub) in zip(kpi_cols, kpis):
    col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Tabs for visuals & predictor
# -------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Cohorts", "Predictor"])

# ---- Tab 1: Overview (compact charts)
with tab1:
    col_a, col_b = st.columns(2)
    # Occupation vs Quality bar
    with col_a:
        st.markdown("#### Avg Sleep Quality by Occupation")
        if 'Occupation' in df_filtered.columns and 'Quality of Sleep' in df_filtered.columns:
            bar_df = (df_filtered.dropna(subset=['Occupation','Quality of Sleep'])
                      .groupby('Occupation')['Quality of Sleep'].mean().reset_index()
                      .sort_values('Quality of Sleep', ascending=False))
            fig_bar = px.bar(bar_df, x='Occupation', y='Quality of Sleep',
                             color='Quality of Sleep', text=bar_df['Quality of Sleep'].round(2),
                             template='plotly_dark', height=340)
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(margin=dict(t=10,b=10,l=10,r=10), xaxis_tickangle=30)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Not enough occupation-quality data to show chart.")
    # Pie and scatter side-by-side
    with col_b:
        st.markdown("#### Sleep Disorder Distribution")
        if 'Sleep Disorder' in df_filtered.columns:
            pie_df = df_filtered['Sleep Disorder'].value_counts().reset_index()
            pie_df.columns = ['Sleep Disorder', 'count']
            fig_pie = px.pie(pie_df, names='Sleep Disorder', values='count', hole=0.35, template='plotly_dark', height=300)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No Sleep Disorder column in dataset")

        st.markdown("#### Stress vs Sleep Duration")
        if 'Stress Level' in df_filtered.columns and 'Sleep Duration' in df_filtered.columns:
            scat_df = df_filtered.dropna(subset=['Stress Level','Sleep Duration'])
            fig_scat = px.scatter(scat_df, x='Stress Level', y='Sleep Duration', color='Quality of Sleep' if 'Quality of Sleep' in df.columns else None,
                                  hover_data=['Person ID','Occupation'] if 'Occupation' in df.columns else None,
                                  template='plotly_dark', height=120)
            fig_scat.update_layout(margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Not enough data for Stress vs Sleep scatter")

# ---- Tab 2: Cohorts and Correlations
with tab2:
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("#### Avg Sleep & SQI by Age Bucket")
        if 'Age Bucket' in df_filtered.columns and 'Sleep Duration' in df_filtered.columns:
            age_df = df_filtered.groupby('Age Bucket').agg({'Sleep Duration':'mean','SQI':'mean','Person ID':'count'}).reset_index().sort_values('Person ID', ascending=False)
            fig_age = px.bar(age_df, x='Age Bucket', y='Sleep Duration', color='SQI', text=age_df['Sleep Duration'].round(2), template='plotly_dark', height=360)
            fig_age.update_layout(margin=dict(t=20,b=20,l=10,r=10))
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Age or Sleep Duration missing for cohort chart")

        st.markdown("#### BMI Category vs Sleep Apnea %")
        if 'BMI Category' in df_filtered.columns and 'Sleep Disorder' in df_filtered.columns:
            bmi_tbl = df_filtered.groupby('BMI Category').apply(lambda d: (d['Sleep Disorder']=='Sleep Apnea').sum()/len(d)*100 if len(d)>0 else 0).reset_index()
            bmi_tbl.columns = ['BMI Category','Apnea %']
            fig_bmi = px.bar(bmi_tbl, x='BMI Category', y='Apnea %', template='plotly_dark', height=200)
            fig_bmi.update_layout(margin=dict(t=10,b=10,l=10,r=10), xaxis_tickangle=25)
            st.plotly_chart(fig_bmi, use_container_width=True)
        else:
            st.info("Not enough BMI + Sleep Disorder data")

    # with c2:
    #     st.markdown("#### Correlation Heatmap (numeric features)")
    #     numeric_df = df_filtered.select_dtypes(include=[np.number])
    #     if numeric_df.shape[1] >= 2:
    #         corr = numeric_df.corr().round(2)
    #         fig_heat = ff.create_annotated_heatmap(
    #             z=corr.values,
    #             x=list(corr.columns),
    #             y=list(corr.index),
    #             annotation_text=corr.values,
    #             colorscale='RdBu', showscale=True
    #         )
    #         fig_heat.update_layout(template='plotly_dark', height=560, margin=dict(t=20,b=20,l=10,r=10))
    #         st.plotly_chart(fig_heat, use_container_width=True)
    #     else:
    #         st.info("Not enough numeric fields for correlations")

# ---- Tab 3: Predictor (train on-the-fly and predict)
with tab3:
    st.markdown("### Sleep Disorder Predictor (trained on current filtered dataset)")
    features = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'Stress Level', 'Heart Rate',
                'Daily Steps', 'Occupation', 'BMI Category']

    df_ml = df.dropna(subset=[c for c in features if c in df.columns] + (['Sleep Disorder'] if 'Sleep Disorder' in df.columns else [])).reset_index(drop=True)
    if 'Sleep Disorder' not in df.columns or df_ml.shape[0] < 10:
        st.warning("Not enough complete rows to train model; predictor disabled. Need at least ~10 complete rows including Sleep Disorder.")
    else:
        # label encode categorical cols present
        label_encoders = {}
        cat_cols = [c for c in ['Gender','Occupation','BMI Category'] if c in df_ml.columns]
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le

        X = df_ml[[c for c in features if c in df_ml.columns]].copy()
        y = df_ml['Sleep Disorder'].astype(str)

        # convert any remaining object columns to numeric codes
        for c in X.columns:
            if X[c].dtype == 'O':
                try:
                    X[c] = pd.to_numeric(X[c])
                except:
                    X[c] = X[c].astype('category').cat.codes

        # train-test split
        try:
            strat = y if y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
        except:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_rows': X_train.shape[0] + X_test.shape[0]
        }

        st.markdown(f"**Trained rows:** {metrics['train_rows']}  &nbsp;&nbsp;  **Accuracy:** {metrics['accuracy']:.3f}  &nbsp;&nbsp; **Precision:** {metrics['precision']:.3f}  &nbsp;&nbsp; **Recall:** {metrics['recall']:.3f}")

        # predictor inputs side-by-side compact
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            age_inp = st.number_input("Age", min_value=0, max_value=120, value=int(df_ml['Age'].median()))
            gender_inp = st.selectbox("Gender", options=sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else ["Unknown"])
            occ_inp = st.selectbox("Occupation", options=sorted(df['Occupation'].dropna().unique().tolist()) if 'Occupation' in df.columns else ["Unknown"])
            bmi_inp = st.selectbox("BMI Category", options=sorted(df['BMI Category'].dropna().unique().tolist()) if 'BMI Category' in df.columns else ["Unknown"])
        with pcol2:
            sd_inp = st.number_input("Sleep Duration (hrs)", min_value=0.0, max_value=24.0, value=float(df_ml['Sleep Duration'].median()))
            sq_inp = st.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=int(df_ml['Quality of Sleep'].median()))
            act_inp = st.number_input("Physical Activity (min/day)", min_value=0, max_value=1440, value=int(df_ml['Physical Activity Level'].median() if 'Physical Activity Level' in df_ml.columns else 30))
            stress_inp = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=int(df_ml['Stress Level'].median() if 'Stress Level' in df_ml.columns else 5))
            hr_inp = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=int(df_ml['Heart Rate'].median() if 'Heart Rate' in df_ml.columns else 65))
            steps_inp = st.number_input("Daily Steps", min_value=0, max_value=100000, value=int(df_ml['Daily Steps'].median() if 'Daily Steps' in df_ml.columns else 3000))

        def safe_encode(le, val):
            try:
                return int(le.transform([val])[0])
            except:
                # unknown -> fallback to 0
                return 0

        if st.button("Predict"):
            # build input vector matching X columns
            input_dict = {}
            for f in X.columns:
                input_dict[f] = np.nan
            # map fields if present
            if 'Age' in X.columns: input_dict['Age'] = age_inp
            if 'Gender' in X.columns: input_dict['Gender'] = safe_encode(label_encoders.get('Gender', LabelEncoder()), str(gender_inp)) if 'Gender' in label_encoders else gender_inp
            if 'Occupation' in X.columns: input_dict['Occupation'] = safe_encode(label_encoders.get('Occupation', LabelEncoder()), str(occ_inp)) if 'Occupation' in label_encoders else occ_inp
            if 'BMI Category' in X.columns: input_dict['BMI Category'] = safe_encode(label_encoders.get('BMI Category', LabelEncoder()), str(bmi_inp)) if 'BMI Category' in label_encoders else bmi_inp
            if 'Sleep Duration' in X.columns: input_dict['Sleep Duration'] = sd_inp
            if 'Quality of Sleep' in X.columns: input_dict['Quality of Sleep'] = sq_inp
            if 'Physical Activity Level' in X.columns: input_dict['Physical Activity Level'] = act_inp
            if 'Stress Level' in X.columns: input_dict['Stress Level'] = stress_inp
            if 'Heart Rate' in X.columns: input_dict['Heart Rate'] = hr_inp
            if 'Daily Steps' in X.columns: input_dict['Daily Steps'] = steps_inp

            X_user = pd.DataFrame([input_dict], columns=X.columns)

            # coerce types similar to training
            for c in X_user.columns:
                if X_user[c].dtype == 'O':
                    try:
                        X_user[c] = pd.to_numeric(X_user[c])
                    except:
                        X_user[c] = X_user[c].astype('category').cat.codes

            try:
                pred = clf.predict(X_user)[0]
                st.success(f"Predicted Sleep Disorder: {pred}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# # -------------------------
# # Footer small notes
# # -------------------------
# st.markdown("---")
# st.markdown("**Notes:** App trains a RandomForest on-the-fly from the CSV (no external model saved). Small datasets or missing columns reduce predictive power. Layout is compact; on small screens some scrolling may remain.")
