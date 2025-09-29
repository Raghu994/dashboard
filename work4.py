import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration - remove all padding and margins
st.set_page_config(
    page_title="Sleep Health & Lifestyle Analysis",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to completely eliminate/minimize scrolling and fix styling
st.markdown("""
<style>
    /* Make the app use full viewport and aggressively reduce spacing so content fits */
    html, body, .stApp {
        height: 100vh;
        overflow: hidden;  /* avoid browser scrollbars */
    }
    .main .block-container {
        padding-top: 0.2rem;
        padding-bottom: 0rem;
        max-width: 99%;
        height: 100vh;
        overflow: hidden; /* avoid internal scrolling as much as possible */
        font-size: 13px;
    }
    .stApp {
        margin-top: -60px;
    }
    /* Compact KPI cards */
    .kpi-card {
        background-color: white;
        padding: 0.45rem;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin: 0.08rem;
        height: 72px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.05rem 0;
    }
    .kpi-title {
        font-size: 0.68rem;
        color: #7f8c8d;
        margin-bottom: 0.05rem;
    }
    .kpi-change {
        font-size: 0.6rem;
        color: #95a5a6;
    }
    /* Insights with black text - compact */
    .insight-item {
        background-color: #f8f9fa;
        padding: 0.45rem;
        border-left: 3px solid #3498db;
        margin: 0.12rem 0;
        border-radius: 3px;
        color: #000000 !important;
        font-size: 0.78rem;
    }
    .insight-item h4, .insight-item p {
        color: #000000 !important;
        margin: 0;
    }
    .insight-item h4 {
        font-size: 0.8rem;
        margin-bottom: 0.12rem;
    }
    /* Prediction results compact */
    .prediction-result {
        padding: 0.6rem;
        border-radius: 6px;
        margin: 0.18rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .risk-none {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .risk-insomnia {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .risk-apnea {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    /* Hide Streamlit branding and reduce spacing */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Remove gaps between elements */
    div[data-testid="stVerticalBlock"] {
        gap: 0.08rem;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    /* Make charts ultra compact */
    .js-plotly-plot .plotly {
        height: 180px !important;
    }
    /* Compact form elements */
    .stNumberInput, .stSelectbox, .stButton {
        margin-bottom: 0.08rem;
        padding-top: 0;
        padding-bottom: 0;
    }
    /* Reduce header sizes */
    h1, h2, h3 {
        margin-bottom: 0.18rem !important;
        margin-top: 0.18rem !important;
    }
    /* Tighter columns on small gaps */
    .css-1lcbmhc.e16nr0p31 { gap: 0.1rem; }
</style>
""", unsafe_allow_html=True)

# Sample dataset - using the exact structure from your CSV
def load_sample_data():
    """Create sample dataset matching your CSV structure"""
    np.random.seed(42)  # For consistent results
    
    data = {
        'Person ID': range(1, 375),
        'Gender': np.random.choice(['Male', 'Female'], 374),
        'Age': np.random.randint(18, 70, 374),
        'Occupation': np.random.choice(['Software Engineer', 'Doctor', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Salesperson', 'Scientist'], 374),
        'Sleep Duration': np.round(np.random.normal(6.8, 1.2, 374), 1),
        'Quality of Sleep': np.random.randint(4, 10, 374),
        'Physical Activity Level': np.random.randint(20, 90, 374),
        'Stress Level': np.random.randint(3, 9, 374),
        'BMI Category': np.random.choice(['Normal', 'Overweight', 'Obese'], 374, p=[0.4, 0.4, 0.2]),
        'Blood Pressure': [f"{np.random.randint(110, 140)}/{np.random.randint(70, 90)}" for _ in range(374)],
        'Heart Rate': np.random.randint(60, 90, 374),
        'Daily Steps': np.random.randint(4000, 12000, 374),
        'Sleep Disorder': np.random.choice(['None', 'Insomnia', 'Sleep Apnea'], 374, p=[0.7, 0.15, 0.15])
    }
    
    # Adjust sleep disorders based on realistic patterns
    df = pd.DataFrame(data)
    
    # More insomnia for high stress and low sleep duration
    insomnia_mask = (df['Stress Level'] > 7) & (df['Sleep Duration'] < 6)
    df.loc[insomnia_mask, 'Sleep Disorder'] = 'Insomnia'
    
    # More sleep apnea for high BMI and older age
    apnea_mask = (df['BMI Category'].isin(['Overweight', 'Obese'])) & (df['Age'] > 40)
    df.loc[apnea_mask, 'Sleep Disorder'] = 'Sleep Apnea'
    
    # More disorders for sedentary occupations - FIXED VERSION
    sedentary_occupations = ['Software Engineer', 'Accountant']
    sedentary_mask = df['Occupation'].isin(sedentary_occupations) & (df['Physical Activity Level'] < 30)
    
    # Get the indices where the mask is True and Sleep Disorder is 'None'
    mask_indices = df[sedentary_mask & (df['Sleep Disorder'] == 'None')].index
    
    if len(mask_indices) > 0:
        # Generate new disorders only for these indices
        new_disorders = np.random.choice(['Insomnia', 'Sleep Apnea'], len(mask_indices), p=[0.6, 0.4])
        df.loc[mask_indices, 'Sleep Disorder'] = new_disorders
    
    return df

class SleepHealthDashboard:
    def __init__(self):
        self.dataset = load_sample_data()
        self.filtered_data = self.dataset.copy()
        
    def update_filters(self, gender_filter, age_filter, bmi_filter):
        """Apply filters to the dataset"""
        self.filtered_data = self.dataset.copy()
        
        # Gender filter
        if gender_filter != "all":
            self.filtered_data = self.filtered_data[self.filtered_data['Gender'] == gender_filter]
        
        # Age filter
        if age_filter != "all":
            if age_filter == "18-30":
                self.filtered_data = self.filtered_data[(self.filtered_data['Age'] >= 18) & (self.filtered_data['Age'] <= 30)]
            elif age_filter == "31-45":
                self.filtered_data = self.filtered_data[(self.filtered_data['Age'] >= 31) & (self.filtered_data['Age'] <= 45)]
            elif age_filter == "46-60":
                self.filtered_data = self.filtered_data[(self.filtered_data['Age'] >= 46) & (self.filtered_data['Age'] <= 60)]
            elif age_filter == "61+":
                self.filtered_data = self.filtered_data[self.filtered_data['Age'] >= 61]
        
        # BMI filter
        if bmi_filter != "all":
            self.filtered_data = self.filtered_data[self.filtered_data['BMI Category'] == bmi_filter]
    
    def calculate_kpis(self):
        """Calculate key performance indicators"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            return {}
        
        kpis = {}
        
        # Average Sleep Duration
        kpis['avg_sleep_duration'] = self.filtered_data['Sleep Duration'].mean()
        
        # Average Sleep Quality
        kpis['avg_sleep_quality'] = self.filtered_data['Quality of Sleep'].mean()
        
        # Sleep Disorders
        sleep_disorders = self.filtered_data['Sleep Disorder'].fillna('None')
        disorder_count = len(sleep_disorders[sleep_disorders != 'None'])
        kpis['disorder_percentage'] = (disorder_count / len(self.filtered_data)) * 100
        kpis['insomnia_count'] = len(sleep_disorders[sleep_disorders == 'Insomnia'])
        kpis['apnea_count'] = len(sleep_disorders[sleep_disorders == 'Sleep Apnea'])
        
        # Physical Activity
        kpis['avg_physical_activity'] = self.filtered_data['Physical Activity Level'].mean()
        
        # Stress Level
        kpis['avg_stress_level'] = self.filtered_data['Stress Level'].mean()
        
        # Heart Rate
        kpis['avg_heart_rate'] = self.filtered_data['Heart Rate'].mean()
        
        return kpis
    
    def create_sleep_duration_quality_chart(self):
        """Create sleep duration vs quality scatter plot"""
        fig = px.scatter(
            self.filtered_data,
            x='Sleep Duration',
            y='Quality of Sleep',
            color='Sleep Disorder',
            color_discrete_map={
                'None': '#2ecc71',
                'Insomnia': '#f39c12',
                'Sleep Apnea': '#e74c3c'
            },
            title='',
            opacity=0.7
        )
        fig.update_layout(
            xaxis_title='Sleep Duration (hours)',
            yaxis_title='Sleep Quality (1-10)',
            height=180,
            showlegend=True,
            margin=dict(l=8, r=8, t=14, b=8)
        )
        return fig
    
    def create_disorders_age_chart(self):
        """Create sleep disorders by age group chart"""
        # Create age groups
        conditions = [
            (self.filtered_data['Age'] <= 30),
            (self.filtered_data['Age'] <= 45) & (self.filtered_data['Age'] > 30),
            (self.filtered_data['Age'] <= 60) & (self.filtered_data['Age'] > 45),
            (self.filtered_data['Age'] > 60)
        ]
        choices = ['18-30', '31-45', '46-60', '61+']
        self.filtered_data['Age Group'] = np.select(conditions, choices, default='Unknown')
        
        # Count disorders by age group
        disorder_counts = pd.crosstab(
            self.filtered_data['Age Group'], 
            self.filtered_data['Sleep Disorder'].fillna('None')
        ).reindex(choices, fill_value=0)
        
        fig = go.Figure()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # None, Insomnia, Sleep Apnea
        
        for i, disorder in enumerate(['None', 'Insomnia', 'Sleep Apnea']):
            if disorder in disorder_counts.columns:
                fig.add_trace(go.Bar(
                    name=disorder,
                    x=disorder_counts.index,
                    y=disorder_counts[disorder],
                    marker_color=colors[i]
                ))
        
        fig.update_layout(
            title='',
            xaxis_title='Age Group',
            yaxis_title='Count',
            barmode='stack',
            height=180,
            showlegend=True,
            margin=dict(l=8, r=8, t=14, b=8)
        )
        return fig
    
    def create_activity_sleep_chart(self):
        """Create physical activity vs sleep quality chart"""
        # Group by activity levels and calculate average sleep quality
        activity_bins = [0, 15, 30, 45, 60, 75, 90, 105]
        activity_labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
        
        self.filtered_data['Activity Group'] = pd.cut(
            self.filtered_data['Physical Activity Level'],
            bins=activity_bins,
            labels=activity_labels,
            right=False
        )
        
        avg_quality_by_activity = self.filtered_data.groupby('Activity Group')['Quality of Sleep'].mean().reset_index()
        
        fig = px.line(
            avg_quality_by_activity,
            x='Activity Group',
            y='Quality of Sleep',
            title='',
            markers=True
        )
        fig.update_layout(
            xaxis_title='Activity (min/day)',
            yaxis_title='Avg Sleep Quality',
            height=180,
            margin=dict(l=8, r=8, t=14, b=8)
        )
        return fig
    
    def create_stress_disorders_chart(self):
        """Create stress level vs sleep disorders chart"""
        # Create stress level groups
        conditions = [
            (self.filtered_data['Stress Level'] <= 3),
            (self.filtered_data['Stress Level'] <= 6) & (self.filtered_data['Stress Level'] > 3),
            (self.filtered_data['Stress Level'] > 6)
        ]
        choices = ['1-3 (Low)', '4-6 (Medium)', '7-10 (High)']
        self.filtered_data['Stress Group'] = np.select(conditions, choices, default='Unknown')
        
        # Create subplots
        stress_groups = ['1-3 (Low)', '4-6 (Medium)', '7-10 (High)']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # None, Insomnia, Sleep Apnea
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f'Stress {group}' for group in stress_groups],
            specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]
        )
        
        for i, stress_group in enumerate(stress_groups):
            group_data = self.filtered_data[self.filtered_data['Stress Group'] == stress_group]
            disorder_counts = group_data['Sleep Disorder'].fillna('None').value_counts()
            
            # Ensure we have all categories
            for disorder in ['None', 'Insomnia', 'Sleep Apnea']:
                if disorder not in disorder_counts:
                    disorder_counts[disorder] = 0
            
            fig.add_trace(go.Pie(
                labels=['None', 'Insomnia', 'Sleep Apnea'],
                values=[disorder_counts.get('None', 0), 
                       disorder_counts.get('Insomnia', 0), 
                       disorder_counts.get('Sleep Apnea', 0)],
                marker_colors=colors,
                hole=0.4,
                showlegend=(i == 0)
            ), 1, i+1)
        
        fig.update_layout(
            title_text='',
            height=180,
            margin=dict(l=8, r=8, t=20, b=8)
        )
        return fig
    
    def calculate_insights(self):
        """Calculate insights from the data"""
        insights = []
        
        # Sleep Duration Impact
        good_sleepers = self.filtered_data[self.filtered_data['Sleep Duration'] >= 7]
        poor_sleepers = self.filtered_data[self.filtered_data['Sleep Duration'] < 6]
        
        if len(good_sleepers) > 0 and len(poor_sleepers) > 0:
            good_disorder_rate = len(good_sleepers[good_sleepers['Sleep Disorder'] != 'None']) / len(good_sleepers)
            poor_disorder_rate = len(poor_sleepers[poor_sleepers['Sleep Disorder'] != 'None']) / len(poor_sleepers)
            reduction = int((1 - good_disorder_rate / poor_disorder_rate) * 100) if poor_disorder_rate > 0 else 0
            insights.append(f"People sleeping 7+ hours have {reduction}% lower risk of sleep disorders.")
        
        # Physical Activity Benefits
        active = self.filtered_data[self.filtered_data['Physical Activity Level'] >= 45]
        inactive = self.filtered_data[self.filtered_data['Physical Activity Level'] < 30]
        
        if len(active) > 0 and len(inactive) > 0:
            active_avg = active['Quality of Sleep'].mean()
            inactive_avg = inactive['Quality of Sleep'].mean()
            improvement = (active_avg - inactive_avg)
            insights.append(f"Active individuals (45+ min/day) report {improvement:.1f} higher sleep quality scores.")
        
        # Stress Management
        high_stress = self.filtered_data[self.filtered_data['Stress Level'] >= 8]
        low_stress = self.filtered_data[self.filtered_data['Stress Level'] <= 4]
        
        if len(high_stress) > 0 and len(low_stress) > 0:
            high_insomnia_rate = len(high_stress[high_stress['Sleep Disorder'] == 'Insomnia']) / len(high_stress)
            low_insomnia_rate = len(low_stress[low_stress['Sleep Disorder'] == 'Insomnia']) / len(low_stress)
            correlation = (high_insomnia_rate / low_insomnia_rate) if low_insomnia_rate > 0 else 0
            insights.append(f"High stress levels correlate with {correlation:.1f}x higher insomnia rates.")
        
        # BMI Correlation
        overweight = self.filtered_data[self.filtered_data['BMI Category'].isin(['Overweight', 'Obese'])]
        normal = self.filtered_data[self.filtered_data['BMI Category'] == 'Normal']
        
        if len(overweight) > 0 and len(normal) > 0:
            overweight_apnea_rate = len(overweight[overweight['Sleep Disorder'] == 'Sleep Apnea']) / len(overweight)
            normal_apnea_rate = len(normal[normal['Sleep Disorder'] == 'Sleep Apnea']) / len(normal)
            correlation = int((overweight_apnea_rate / normal_apnea_rate - 1) * 100) if normal_apnea_rate > 0 else 0
            insights.append(f"Overweight individuals show {correlation}% higher sleep apnea prevalence.")
        
        return insights
    
    def parse_blood_pressure(self, bp_string):
        """Parse blood pressure string into systolic and diastolic"""
        try:
            systolic, diastolic = map(int, bp_string.split('/'))
            return systolic, diastolic
        except:
            return 120, 80  # Default normal values
    
    def predict_sleep_disorder(self, age, gender, occupation, sleep_duration, sleep_quality, 
                             activity, stress, bmi, blood_pressure, heart_rate, daily_steps):
        """COMPREHENSIVE sleep disorder prediction using ALL input parameters"""
        
        risk_score = 0
        reasons = []
        
        # 1. Age factor (higher risk for older adults)
        if age > 60:
            risk_score += 3
            reasons.append("Age over 60 significantly increases risk")
        elif age > 50:
            risk_score += 2
            reasons.append("Age over 50 increases risk")
        elif age > 40:
            risk_score += 1
            reasons.append("Age over 40 slightly increases risk")
        
        # 2. Gender factor (males have higher sleep apnea risk)
        if gender == 'Male':
            risk_score += 1
            reasons.append("Male gender increases sleep apnea risk")
        
        # 3. Occupation factors
        high_stress_occupations = ['Doctor', 'Nurse', 'Software Engineer', 'Salesperson']
        sedentary_occupations = ['Software Engineer', 'Accountant', 'Scientist']
        
        if occupation in high_stress_occupations:
            risk_score += 2
            reasons.append(f"High-stress occupation ({occupation})")
        
        if occupation in sedentary_occupations:
            risk_score += 1
            reasons.append(f"Sedentary occupation ({occupation})")
        
        # 4. Sleep duration factors
        if sleep_duration < 5:
            risk_score += 3
            reasons.append("Severely insufficient sleep duration")
        elif sleep_duration < 6:
            risk_score += 2
            reasons.append("Insufficient sleep duration")
        elif sleep_duration > 9:
            risk_score += 2
            reasons.append("Excessive sleep duration")
        
        # 5. Sleep quality
        if sleep_quality < 4:
            risk_score += 3
            reasons.append("Very poor sleep quality")
        elif sleep_quality < 6:
            risk_score += 2
            reasons.append("Poor sleep quality")
        
        # 6. Physical activity
        if activity < 20:
            risk_score += 2
            reasons.append("Very low physical activity")
        elif activity < 30:
            risk_score += 1
            reasons.append("Low physical activity")
        
        # 7. Stress level
        if stress > 8:
            risk_score += 3
            reasons.append("Very high stress level")
        elif stress > 7:
            risk_score += 2
            reasons.append("High stress level")
        
        # 8. BMI category
        if bmi == 'Obese':
            risk_score += 3
            reasons.append("Obese BMI significantly increases risk")
        elif bmi == 'Overweight':
            risk_score += 2
            reasons.append("Overweight BMI increases risk")
        
        # 9. Blood Pressure analysis
        systolic, diastolic = self.parse_blood_pressure(blood_pressure)
        if systolic >= 140 or diastolic >= 90:
            risk_score += 3
            reasons.append("High blood pressure (Hypertension)")
        elif systolic >= 130 or diastolic >= 85:
            risk_score += 2
            reasons.append("Elevated blood pressure")
        
        # 10. Heart rate
        if heart_rate > 90:
            risk_score += 2
            reasons.append("Elevated resting heart rate")
        elif heart_rate > 85:
            risk_score += 1
            reasons.append("Slightly elevated heart rate")
        
        # 11. Daily steps (activity level)
        if daily_steps < 5000:
            risk_score += 2
            reasons.append("Sedentary lifestyle (low daily steps)")
        elif daily_steps < 7000:
            risk_score += 1
            reasons.append("Below recommended daily steps")
        
        # Determine final risk level and disorder type
        if risk_score >= 10:
            # High risk - determine specific disorder
            if bmi in ['Overweight', 'Obese'] and (systolic >= 130 or sleep_duration > 8):
                risk_level = 'Sleep Apnea'
                confidence = 'Very High'
            else:
                risk_level = 'Insomnia'
                confidence = 'Very High'
        elif risk_score >= 7:
            risk_level = 'Insomnia'
            confidence = 'High'
        elif risk_score >= 5:
            risk_level = 'Insomnia'
            confidence = 'Medium'
        elif risk_score >= 3:
            risk_level = 'None'
            confidence = 'Low'
            reasons = ["Low risk profile based on current factors"]
        else:
            risk_level = 'None'
            confidence = 'Very Low'
            reasons = ["Excellent sleep health profile"]
        
        return risk_level, confidence, reasons, risk_score

def main():
    # Initialize dashboard
    dashboard = SleepHealthDashboard()
    
    # Ultra-compact header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("<h1 style='margin-bottom: 0;'>🌙 Sleep Health & Lifestyle Analysis</h1>", unsafe_allow_html=True)
    with col2:
        st.write(f"**Records:** {len(dashboard.dataset)}")
    with col3:
        st.write(f"**Updated:** {datetime.now().strftime('%H:%M')}")
    
    # Ultra-compact filters
    st.markdown("<h3 style='margin-bottom: 0.2rem;'>Filters</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        gender_options = ['all'] + list(dashboard.dataset['Gender'].unique())
        gender_filter = st.selectbox("Gender", gender_options, label_visibility="collapsed")
    
    with col2:
        age_filter = st.selectbox("Age Range", ['all', '18-30', '31-45', '46-60', '61+'], label_visibility="collapsed")
    
    with col3:
        bmi_options = ['all'] + list(dashboard.dataset['BMI Category'].unique())
        bmi_filter = st.selectbox("BMI Category", bmi_options, label_visibility="collapsed")
    
    with col4:
        apply_clicked = st.button("Apply Filters", use_container_width=True)
    
    with col5:
        if st.button("Reset Filters", use_container_width=True):
            dashboard.update_filters('all', 'all', 'all')
            st.rerun()
    
    # Apply filters
    if apply_clicked:
        dashboard.update_filters(gender_filter, age_filter, bmi_filter)
    else:
        dashboard.update_filters('all', 'all', 'all')
    
    # Ultra-compact KPIs
    st.markdown("<h3 style='margin-bottom: 0.2rem;'>Key Metrics</h3>", unsafe_allow_html=True)
    kpis = dashboard.calculate_kpis()
    
    if kpis:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metrics = [
            (f"{kpis['avg_sleep_duration']:.1f} hrs", "Sleep Duration", "+0.2 vs recommended" if kpis['avg_sleep_duration'] >= 7 else "Below recommended"),
            (f"{kpis['avg_sleep_quality']:.1f}/10", "Sleep Quality", "Good" if kpis['avg_sleep_quality'] >= 7 else "Needs improvement"),
            (f"{kpis['disorder_percentage']:.0f}%", "Sleep Disorders", f"{kpis['insomnia_count']} Insomnia, {kpis['apnea_count']} Apnea"),
            (f"{kpis['avg_physical_activity']:.0f} min", "Physical Activity", "+7 min vs target" if kpis['avg_physical_activity'] >= 30 else "Below target"),
            (f"{kpis['avg_stress_level']:.1f}/10", "Stress Level", "Low" if kpis['avg_stress_level'] <= 5 else "Moderate"),
            (f"{kpis['avg_heart_rate']:.0f} bpm", "Heart Rate", "60-100 bpm normal")
        ]
        
        for i, (value, title, change) in enumerate(metrics):
            with [col1, col2, col3, col4, col5, col6][i]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-change">{change}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tabs for visuals + insights + predictor
    tab1, tab2, tab3 = st.tabs(["Sleep Analysis", "Key Insights", "Risk Assessment"])
    
    # Ultra-compact charts
    with tab1:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Sleep Analysis</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = dashboard.create_sleep_duration_quality_chart()
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
            
            fig3 = dashboard.create_activity_sleep_chart()
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig2 = dashboard.create_disorders_age_chart()
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            
            fig4 = dashboard.create_stress_disorders_chart()
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
    
    # Ultra-compact insights
    with tab2:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Key Insights</h3>", unsafe_allow_html=True)
        insights = dashboard.calculate_insights()
        
        if insights:
            cols = st.columns(2)
            for i, insight in enumerate(insights):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="insight-item">
                        <h4>Insight {i+1}</h4>
                        <p>{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Comprehensive prediction with ALL parameters
    with tab3:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Sleep Disorder Risk Assessment</h3>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Row 1
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                predict_age = st.number_input("Age", min_value=18, max_value=100, value=35)
                predict_occupation = st.selectbox("Occupation", dashboard.dataset['Occupation'].unique())
            
            with col2:
                predict_gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
                predict_sleep_duration = st.number_input("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
            
            with col3:
                predict_sleep_quality = st.number_input("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)
                predict_activity = st.number_input("Physical Activity (min/day)", min_value=0, max_value=180, value=45)
            
            with col4:
                predict_stress = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5)
                predict_bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"], key="pred_bmi")
            
            # Row 2
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                predict_heartrate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=72)
            
            with col6:
                predict_bp_systolic = st.number_input("BP Systolic", min_value=90, max_value=200, value=120)
                predict_bp_diastolic = st.number_input("BP Diastolic", min_value=60, max_value=130, value=80)
            
            with col7:
                predict_daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=20000, value=8000)
            
            with col8:
                predict_button = st.form_submit_button("🔍 Assess Sleep Disorder Risk", use_container_width=True)
            
            if predict_button:
                blood_pressure = f"{predict_bp_systolic}/{predict_bp_diastolic}"
                risk_level, confidence, reasons, risk_score = dashboard.predict_sleep_disorder(
                    predict_age, predict_gender, predict_occupation, predict_sleep_duration, 
                    predict_sleep_quality, predict_activity, predict_stress, predict_bmi, 
                    blood_pressure, predict_heartrate, predict_daily_steps
                )
                
                # Display risk score
                st.write(f"**Risk Score:** {risk_score}/20")
                
                if risk_level == 'None':
                    st.markdown(f"""
                    <div class="prediction-result risk-none">
                        <p>✅ Low Risk of Sleep Disorders ({confidence})</p>
                        <p>Your sleep habits appear healthy. Continue maintaining good sleep hygiene.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == 'Insomnia':
                    reasons_text = ' • '.join(reasons)
                    st.markdown(f"""
                    <div class="prediction-result risk-insomnia">
                        <p>⚠️ Potential Risk of Insomnia ({confidence} Confidence)</p>
                        <p><strong>Factors:</strong> {reasons_text}</p>
                        <p style="margin-top: 8px; font-weight: normal;">Consider improving sleep duration, managing stress, and increasing physical activity.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    reasons_text = ' • '.join(reasons)
                    st.markdown(f"""
                    <div class="prediction-result risk-apnea">
                        <p>🚨 Potential Risk of Sleep Apnea ({confidence} Confidence)</p>
                        <p><strong>Factors:</strong> {reasons_text}</p>
                        <p style="margin-top: 8px; font-weight: normal;">Consult a healthcare provider for sleep study evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()