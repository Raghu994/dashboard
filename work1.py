# dashboard.py (compact dashboard layout)
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

# --- Title
st.title("Sleep Health Dashboard")

# --- Required columns
REQUIRED_COLS = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps',
    'BMI Category', 'Sleep Disorder'
]

@st.cache_data
def load_data(path="Sleep_health_and_lifestyle_dataset.csv"):
    return pd.read_csv(path)

# Load
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Make sure the CSV is in the folder.")
    st.stop()

# Check columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# Coerce numeric
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'Stress Level',
                'Heart Rate', 'Daily Steps']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- Sidebar filters
st.sidebar.header("Filters")
selected_gender = st.sidebar.selectbox(
    "Gender", options=['All'] + sorted(df['Gender'].dropna().unique().tolist())
)
selected_occupation = st.sidebar.selectbox(
    "Occupation", options=['All'] + sorted(df['Occupation'].dropna().unique().tolist())
)

df_filtered = df.copy()
if selected_gender != 'All':
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
if selected_occupation != 'All':
    df_filtered = df_filtered[df_filtered['Occupation'] == selected_occupation]

# --- KPIs Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Participants", len(df_filtered))
if len(df_filtered) > 0:
    col2.metric("Avg Sleep (hrs)", f"{df_filtered['Sleep Duration'].mean():.1f}")
    col3.metric("Avg Quality", f"{df_filtered['Quality of Sleep'].mean():.1f}")
    col4.metric("Avg Stress", f"{df_filtered['Stress Level'].mean():.1f}")
else:
    col2.metric("Avg Sleep (hrs)", "N/A")
    col3.metric("Avg Quality", "N/A")
    col4.metric("Avg Stress", "N/A")

# --- Tabs for visuals + predictor
tab1, tab2, tab3 = st.tabs(["Occupations", "Correlations", "Predictor"])

# ---- Tab 1: Occupation vs Sleep Quality
with tab1:
    if df_filtered.empty:
        st.warning("No data to plot.")
    else:
        avg_quality_by_occ = (
            df_filtered.dropna(subset=['Occupation', 'Quality of Sleep'])
            .groupby('Occupation')['Quality of Sleep']
            .mean()
            .reset_index()
        )
        if not avg_quality_by_occ.empty:
            fig_bar = px.bar(
                avg_quality_by_occ, x='Occupation', y='Quality of Sleep',
                title="Average Sleep Quality by Occupation",
                color='Quality of Sleep', color_continuous_scale='Blues',
                text='Quality of Sleep'
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Not enough occupation-quality data to show chart.")

# ---- Tab 2: Correlation Heatmap
with tab2:
    numerical_df = df_filtered.select_dtypes(include=[np.number])
    if numerical_df.shape[1] < 2:
        st.warning("Not enough numeric features for correlations.")
    else:
        corr = numerical_df.corr().round(2)
        fig_heatmap = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.values,
            colorscale='RdBu', showscale=True
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# ---- Tab 3: Prediction Tool
with tab3:
    st.subheader("Sleep Disorder Predictor")

    features = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'Stress Level', 'Heart Rate',
                'Daily Steps', 'Occupation', 'BMI Category']

    df_ml = df.dropna(subset=features + ['Sleep Disorder']).reset_index(drop=True)
    if df_ml.empty:
        st.error("Not enough complete rows to train model.")
    else:
        # Label encode
        label_encoders = {}
        cat_cols = ['Gender', 'Occupation', 'BMI Category']
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le

        X = df_ml[features].copy()
        for c in cat_cols:
            X[c] = X[c].astype(int)
        y = df_ml['Sleep Disorder'].astype(str)

        @st.cache_resource
        def train_model(X, y, random_state=42):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=random_state, stratify=y
            )
            model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return model, {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
            }

        model, metrics = train_model(X, y)

        st.write(f"**Accuracy:** {metrics['accuracy']:.3f} | "
                 f"**Precision:** {metrics['precision']:.3f} | "
                 f"**Recall:** {metrics['recall']:.3f}")

        # Input UI (side-by-side)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", int(df_ml['Age'].min()), int(df_ml['Age'].max()), int(df_ml['Age'].median()))
            gender = st.selectbox("Gender", options=sorted(df['Gender'].dropna().unique()))
            occ = st.selectbox("Occupation", options=sorted(df['Occupation'].dropna().unique()))
            bmi = st.selectbox("BMI Category", options=sorted(df['BMI Category'].dropna().unique()))
        with c2:
            sd = st.slider("Sleep Duration (hrs)", float(df_ml['Sleep Duration'].min()), float(df_ml['Sleep Duration'].max()), float(df_ml['Sleep Duration'].median()))
            sq = st.slider("Quality of Sleep (1-10)", 1, 10, int(df_ml['Quality of Sleep'].median()))
            act = st.slider("Physical Activity (min/day)", int(df_ml['Physical Activity Level'].min()), int(df_ml['Physical Activity Level'].max()), int(df_ml['Physical Activity Level'].median()))
            stress = st.slider("Stress Level (1-10)", 1, 10, int(df_ml['Stress Level'].median()))
            hr = st.slider("Heart Rate (bpm)", int(df_ml['Heart Rate'].min()), int(df_ml['Heart Rate'].max()), int(df_ml['Heart Rate'].median()))
            steps = st.slider("Daily Steps", int(df_ml['Daily Steps'].min()), int(df_ml['Daily Steps'].max()), int(df_ml['Daily Steps'].median()))

        def safe_encode(le, val):
            try:
                return int(le.transform([val])[0])
            except:
                return 0

        if st.button("Predict"):
            gender_code = safe_encode(label_encoders['Gender'], str(gender))
            occ_code = safe_encode(label_encoders['Occupation'], str(occ))
            bmi_code = safe_encode(label_encoders['BMI Category'], str(bmi))

            user_input = np.array([
                age, gender_code, sd, sq, act, stress, hr, steps, occ_code, bmi_code
            ]).reshape(1, -1)

            pred = model.predict(user_input)[0]
            st.success(f"Predicted Sleep Disorder: {pred}")
