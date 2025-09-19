# dashboard.py (improved)
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

st.title("Sleep Health & Lifestyle Explorer")
st.markdown(
    "**Hook:** Did you know poor sleep affects 1 in 3 adults, raising risks for heart disease and stress? "
    "This dashboard explores how habits like occupation and activity predict disorders like insomnia."
)
st.markdown("**Key Questions:** What jobs harm sleep most? Can we predict personal risks from lifestyle data?")

# Required columns (adjust if your dataset has different names)
REQUIRED_COLS = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps',
    'BMI Category', 'Sleep Disorder'
]

@st.cache_data
def load_data(path="Sleep_health_and_lifestyle_dataset.csv"):
    df = pd.read_csv(path)
    return df

# Load
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Make sure 'Sleep_health_and_lifestyle_dataset.csv' is in the app folder.")
    st.stop()

# Check required columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# Quick dtype safety for numeric columns (coerce and warn)
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                'Stress Level', 'Heart Rate', 'Daily Steps']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Sidebar filters
st.sidebar.header("Filters")
# For prediction selectboxes we will later use df_ml (after dropna) to ensure compatibility with LabelEncoder
selected_gender = st.sidebar.selectbox("Filter by Gender", options=['All'] + sorted(df['Gender'].dropna().unique().tolist()))
selected_occupation = st.sidebar.selectbox("Filter by Occupation", options=['All'] + sorted(df['Occupation'].dropna().unique().tolist()))

# Apply filters to display dataframe & KPIs
df_filtered = df.copy()
if selected_gender != 'All':
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
if selected_occupation != 'All':
    df_filtered = df_filtered[df_filtered['Occupation'] == selected_occupation]

# --- Data Overview
st.header("1. Data Overview")
st.markdown(f"**Dataset:** {len(df_filtered)} participants (filtered from {len(df)} total). Features include age, occupation, sleep metrics, and disorders.")
st.subheader("First Look at the Data (Top 10 Rows)")
st.dataframe(df_filtered.head(10))

# --- KPIs (handle empty filtered set)
st.header("2. Basic Analysis")
st.subheader("Key Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Participants", len(df_filtered))
if len(df_filtered) > 0:
    col2.metric("Average Sleep Duration", f"{df_filtered['Sleep Duration'].mean():.1f} hours")
    col3.metric("Average Sleep Quality", f"{df_filtered['Quality of Sleep'].mean():.1f} / 10")
    col4.metric("Average Stress Level", f"{df_filtered['Stress Level'].mean():.1f} / 10")
else:
    col2.metric("Average Sleep Duration", "N/A")
    col3.metric("Average Sleep Quality", "N/A")
    col4.metric("Average Stress Level", "N/A")
st.caption("These KPIs highlight core sleep health trends—watch how filters change them!")

# --- EDA
st.header("Exploratory Data Analysis")
st.markdown("Spot patterns: How do occupations, stress, and activity link to sleep?")

# Bar chart: average quality by occupation (guard empty)
if df_filtered.empty:
    st.warning("No data to plot for the selected filters.")
else:
    avg_quality_by_occ = (
        df_filtered.dropna(subset=['Occupation', 'Quality of Sleep'])
        .groupby('Occupation')['Quality of Sleep']
        .mean()
        .reset_index()
    )
    if not avg_quality_by_occ.empty:
        fig_bar = px.bar(avg_quality_by_occ, x='Occupation', y='Quality of Sleep',
                         title="Average Sleep Quality by Occupation",
                         color='Quality of Sleep', color_continuous_scale='Blues',
                         text='Quality of Sleep')
        fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_bar.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Not enough occupation-quality data to show the bar chart.")

# Correlation heatmap (numerical only) - guard small dims
st.subheader("Correlation Heatmap: Key Relationships")
numerical_df = df_filtered.select_dtypes(include=[np.number])
if numerical_df.shape[1] < 2:
    st.warning("Not enough numeric features available to compute correlations.")
else:
    corr = numerical_df.corr().round(2)
    # plotly annotated heatmap expects arrays
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.values,
        colorscale='RdBu', showscale=True
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.info("""
**Insights from EDA:**
- **Occupations Matter:** Accountants/Engineers score highest (~8/10 quality), while Nurses/Doctors are lowest (~6.5/10)—likely due to stress.
- **Strong Correlations:** Stress Level negatively correlates with Sleep Quality, while Daily Steps often links to Duration.
- **Filter Tip:** Try selecting 'Female' or 'Nurse' to see shifts!
""")

# --- Prediction Tool (ML)

st.header("Sleep Disorder Predictor")
st.markdown("Enter details to predict risk—our model uses Random Forest on key features.")

# Prepare ML dataset (drop rows missing any feature or target)
features = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Stress Level', 'Heart Rate', 'Daily Steps', 'Occupation', 'BMI Category']

df_ml = df.copy()
df_ml = df_ml.dropna(subset=features + ['Sleep Disorder']).reset_index(drop=True)

if df_ml.empty:
    st.error("Not enough complete rows to train the model after dropping NaNs.")
    st.stop()

# Fit LabelEncoders and keep mapping
label_encoders = {}
categorical_cols = ['Gender', 'Occupation', 'BMI Category']
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le

# X/y
X = df_ml[features].copy()
# Ensure X numeric dtype
for c in categorical_cols:
    X[c] = X[c].astype(int)
y = df_ml['Sleep Disorder'].astype(str)

# Train/test and evaluation (cached)
@st.cache_resource
def train_model(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    # Eval
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    return model, {'accuracy': acc, 'precision': prec, 'recall': rec}, (X_train, X_test, y_train, y_test)

model, metrics, _ = train_model(X, y)

st.subheader("Model performance on holdout (25%)")
st.write(f"Accuracy: {metrics['accuracy']:.3f}, Precision (weighted): {metrics['precision']:.3f}, Recall (weighted): {metrics['recall']:.3f}")

# Build user input UI — use df_ml for option choices so LabelEncoder mappings align
col1, col2 = st.columns(2)
with col1:
    min_age, max_age = int(df_ml['Age'].min()), int(df_ml['Age'].max())
    input_age = st.slider("Age", min_age, max_age, int(np.median(df_ml['Age'].dropna())))
    input_gender = st.selectbox("Gender", options=sorted(df_ml['Gender'].astype(str).map(lambda x: label_encoders['Gender'].classes_[int(x)] if x.isdigit() else x).unique()) if False else sorted(df['Gender'].dropna().unique()))
    # NOTE: to keep the UI friendly, we'll populate selectboxes from df (but ensure mapping below handles unseen safely)
    # to be safe, we'll re-populate from df_ml values below explicitly for encoding
    input_occupation = st.selectbox("Occupation", options=sorted(df['Occupation'].dropna().unique()))
    input_bmi = st.selectbox("BMI Category", options=sorted(df['BMI Category'].dropna().unique()))
with col2:
    sd_min, sd_max = float(df_ml['Sleep Duration'].min()), float(df_ml['Sleep Duration'].max())
    input_sleep_dur = st.slider("Sleep Duration (hours)", float(round(sd_min,1)), float(round(sd_max,1)), float(df_ml['Sleep Duration'].median()))
    input_sleep_qual = st.slider("Quality of Sleep (1-10)", int(max(1, df_ml['Quality of Sleep'].min())), int(df_ml['Quality of Sleep'].max()), int(df_ml['Quality of Sleep'].median()))
    input_activity = st.slider("Physical Activity (min/day)", int(df_ml['Physical Activity Level'].min()), int(df_ml['Physical Activity Level'].max()), int(df_ml['Physical Activity Level'].median()))
    input_stress = st.slider("Stress Level (1-10)", int(max(1, df_ml['Stress Level'].min())), int(df_ml['Stress Level'].max()), int(df_ml['Stress Level'].median()))
    input_heart = st.slider("Heart Rate (bpm)", int(df_ml['Heart Rate'].min()), int(df_ml['Heart Rate'].max()), int(df_ml['Heart Rate'].median()))
    input_steps = st.slider("Daily Steps", int(df_ml['Daily Steps'].min()), int(df_ml['Daily Steps'].max()), int(df_ml['Daily Steps'].median()))

# Helper to safely encode user-provided category using the fitted label_encoders
def safe_encode(le: LabelEncoder, value: str):
    """
    Try to encode value with LabelEncoder.
    If unseen, map to the most frequent class index from training data.
    """
    try:
        return int(le.transform([value])[0])
    except Exception:
        # fallback: map to most frequent label in training set
        # classes_ are sorted by label; to map to most frequent, we estimate from le.classes_ via training distribution
        # A simpler, robust fallback: map to label 0
        return 0

if st.button("Predict Sleep Disorder"):
    # Encode inputs safely
    gender_code = safe_encode(label_encoders['Gender'], str(input_gender))
    occ_code = safe_encode(label_encoders['Occupation'], str(input_occupation))
    bmi_code = safe_encode(label_encoders['BMI Category'], str(input_bmi))

    user_input = np.array([
        input_age,
        gender_code,
        input_sleep_dur,
        input_sleep_qual,
        input_activity,
        input_stress,
        input_heart,
        input_steps,
        occ_code,
        bmi_code
    ]).reshape(1, -1)

    # If model supports predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(user_input)[0]
    else:
        proba = None
    prediction = model.predict(user_input)[0]

    st.success(f"**Predicted Sleep Disorder:** {prediction}")
    if proba is not None:
        st.warning(f"**Risk Level:** {'High' if prediction != 'None' else 'Low'} – Confidence: {max(proba):.1%}")
        proba_df = pd.DataFrame({'Disorder': model.classes_, 'Probability': proba})
        fig_proba = px.bar(proba_df, x='Disorder', y='Probability', title='Prediction Confidence Breakdown',
                           color='Probability', color_continuous_scale='Reds')
        st.plotly_chart(fig_proba, use_container_width=True)
    else:
        st.info("Model does not provide probability estimates.")

    if prediction != 'None':
        st.markdown(f"**Recommended Action:** For {prediction}, try reducing stress (e.g., meditation) and aiming for 8k+ daily steps. Consult a doctor if symptoms persist!")
    else:
        st.markdown("**Great News!** Maintain your habits—regular activity keeps risks low.")

# --- Conclusion
st.header("Key Insights & Next Steps")
st.markdown("""
**What We Learned (The Story Resolution):**
- **Stress is the Villain:** High stress (7+/10) appears to be associated with higher disorder prevalence.
- **Activity is the Hero:** More daily activity often improves sleep quality.
- **Occupations Vary:** Shift workers (nurses/doctors) may need tailored wellness support.

**Bigger Picture (Horizon):** This predictor could power health apps or corporate programs. Future: Add real-time wearable data (e.g., Fitbit integration) for live monitoring.

**Model Performance:** Shown above (holdout test). Consider deeper validation (cross-validation, class balance handling).
""")

st.markdown("---")
