import streamlit as st
import pandas as pd
import time
import joblib
import numpy as np
from datetime import datetime
from catboost import CatBoostClassifier
import shap
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


st.set_page_config(page_title="SDIN Security Monitor", layout="wide")

# ZSAL and SOAM functions
def generate_zasl_label(shap_values, feature_names):
    positive_contributions = [(val, name) for val, name in zip(shap_values, feature_names) if val > 0]
    sorted_contributions = sorted(positive_contributions, key=lambda x: x[0], reverse=True)

    top_features = [item[1] for item in sorted_contributions[:3]]
    high_impact_extra = [item[1] for item in sorted_contributions[3:] if item[0] > 2]

    final_feature_list = top_features + high_impact_extra
    if not top_features:
        return "Generic Anomaly"

    return f"{'-'.join(final_feature_list)}-based-Attribution"

def calculate_weighted_soam(instance_shap, mean_profile):
    # 1. Cosine Similarity (Weight: 70%)
    cos_sim = 1 - cosine(instance_shap, mean_profile)
    # 2. Spearman Rank Correlation (Weight: 10%)
    spearman_corr, _ = spearmanr(instance_shap, mean_profile)
    # 3. Absolute Difference (Weight: 10%)
    mad_score = np.mean(np.abs(instance_shap - mean_profile))
    # 4. Sign Overlap (Weight: 10%)
    sign_overlap = (np.sign(instance_shap) == np.sign(mean_profile)).mean()

    weighted_score = (0.70 * cos_sim) + (0.1 * spearman_corr) + (0.1 * sign_overlap) + (0.1 * mad_score)

    return {"Weighted_Reliability": weighted_score, "Cosine": cos_sim}

def get_soam_verdict(instance_shap, tp_mean, fp_mean):
    tp_rel = calculate_weighted_soam(instance_shap, tp_mean)['Weighted_Reliability']
    fp_rel = calculate_weighted_soam(instance_shap, fp_mean)['Weighted_Reliability']

    if tp_rel > fp_rel and (tp_rel - fp_rel) > 0.15:
        verdict, action = "CONFIRMED ATTACK", "Display in System"
    elif fp_rel > tp_rel:
        verdict, action = "SUSPECT Attack (Potential False Positive)", "LOW - For Analyst Review"
    else:
        verdict, action = "SUSPECT Attack (Potential True Positive)", "HIGH - For Analyst Review"

    return {"Verdict": verdict, "Action": action, "TrueProfile_Similarity": tp_rel, "FalseProfile_Similarity": fp_rel}

def get_soam_verdict_NormalValidation(instance_shap, normal_mean, fn_mean, feature_names):
    normal_sim = calculate_weighted_soam(instance_shap, normal_mean)['Weighted_Reliability']
    fn_sim = calculate_weighted_soam(instance_shap, fn_mean)['Weighted_Reliability']

    if fn_sim > normal_sim and fn_sim > 0.4:
        label = generate_zasl_label(instance_shap, feature_names)
        return {"Label": label, "Verdict": "SUSPECT Attack (Potential False Negative)", 
                "Action": "MEDIUM - For Analyst Review", "Normal_Sim": normal_sim, "FN_Sim": fn_sim}
    
    return {"Label": "Normal Traffic", "Verdict": "NORMAL Traffic", 
            "Action": "No Action Required", "Normal_Sim": normal_sim, "FN_Sim": fn_sim}

# DATA LOADING
@st.cache_resource
def load_assets():
    model = CatBoostClassifier()
    model.load_model('res/catboost_sdn_20260123_040438.cbm')
    
    X_test = joblib.load('res/Data_Splits/X_test.pkl')
    feature_names = ["proto_number", "Dur", "Mean", "Stddev", "Min", "Max", "Pkts", "Bytes",
                     "Spkts", "Dpkts", "Sbytes", "Dbytes", "Srate", "Drate", "Sum",
                     "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP", "TnP_PerProto",
                     "TnP_Per_Dport", "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP"]
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    profiles = {
        'tp_mean': joblib.load('res/SOAM_Baselines2/tp_mean_profile_20260325_115545.pkl'),
        'fp_mean': joblib.load('res/SOAM_Baselines2/fp_mean_profile_20260325_115545.pkl'),
        'normal_mean': joblib.load('res/SOAM_Baselines2/normal_mean_profile_20260325_115545.pkl'),
        'fn_mean': joblib.load('res/SOAM_Baselines2/fn_mean_profile_20260325_115545.pkl')
    }
    explainer = shap.TreeExplainer(model)
    return model, X_test_df, profiles, explainer

# UI
st.title("🛡️ SDN Security Monitor - X-IDS Protocol")
st.markdown("### Explainable Zero-Shot Attack Attribution in Software-Defined Industrial Networks")

col1, col2, col3 = st.columns(3)
total_packets_metric = col1.metric("Packets Processed", "0")
alerts_found_metric = col2.metric("Attacks Detected", "0")
system_status = col3.metric("System Health", "Active")

st.divider()

if st.button("Start Live Network Simulation"):
    model, X_test_df, profiles, explainer = load_assets()
    table_placeholder = st.empty()
    log_data = []
    samples = X_test_df.sample(20)
    attack_count = 0
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        packet_df = pd.DataFrame([row])
        pred = model.predict(packet_df)[0]
        
        shap_vals = explainer.shap_values(packet_df)
        if isinstance(shap_vals, list): shap_vals = shap_vals[0]
        shap_vals = shap_vals.flatten()

        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if pred == 1:
            attack_count += 1
            label = generate_zasl_label(shap_vals, X_test_df.columns.tolist())
            v = get_soam_verdict(shap_vals, profiles['tp_mean'], profiles['fp_mean'])
            icon = "🔴" if "CONFIRMED" in v['Verdict'] else "🟡"
            status, detail = f"{icon} {v['Verdict']}", f"[{label}] -> {v['Action']}"
        else:
            v = get_soam_verdict_NormalValidation(shap_vals, profiles['normal_mean'], profiles['fn_mean'], X_test_df.columns.tolist())
            icon = "🟠" if "SUSPECT" in v['Verdict'] else "🟢"
            status, detail = f"{icon} {v['Verdict']}", f"[{v['Label']}] -> {v['Action']}"

        log_data.insert(0, {"Timestamp": timestamp, "Result": status, "Details": detail})
        
        table_placeholder.table(pd.DataFrame(log_data))
        total_packets_metric.metric("Packets Processed", i + 1)
        alerts_found_metric.metric("Attacks Detected", attack_count)
        time.sleep(0.6)

    st.success("Simulation Complete")