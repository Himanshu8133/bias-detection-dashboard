# ==========================================
# Bias Detection Dashboard
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="Bias Detection Dashboard",
    page_icon="",
    layout="wide"
)

st.title("Dataset Bias Checker")
st.write(
    "This system performs data-level bias analysis and provides precise "
    "technical recommendations indicating where and how the dataset should be corrected."
)

# ------------------------------------------
# BACKEND LOGIC
# ------------------------------------------
def bias_analysis(df, sensitive, target):
    report = {}

    # 🔧 Ensure target is numeric (FIX for TypeError)
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = pd.factorize(df[target])[0]
        report["target_encoded"] = True
    else:
        report["target_encoded"] = False

    report["rows"] = len(df)
    report["columns"] = len(df.columns)

    # Representation bias
    rep = df[sensitive].value_counts(normalize=True)
    dominant_group = rep.idxmax()
    dominant_ratio = rep.max()
    rep_issue = dominant_ratio > 0.7

    # Label bias
    label_dist = df.groupby(sensitive)[target].mean()
    diff = label_dist.max() - label_dist.min()
    label_issue = diff > 0.2

    # Fairness metrics
    spd = diff
    di = label_dist.min() / label_dist.max() if label_dist.max() > 0 else 1
    fairness_issue = spd > 0.1 or di < 0.8

    issues = sum([rep_issue, label_issue, fairness_issue])
    traffic = "green" if issues == 0 else "yellow" if issues == 1 else "red"

    report.update({
        "rep": rep,
        "label_dist": label_dist,
        "rep_issue": rep_issue,
        "label_issue": label_issue,
        "fairness_issue": fairness_issue,
        "traffic": traffic,
        "dominant_group": dominant_group,
        "dominant_ratio": dominant_ratio,
        "spd": spd,
        "di": di
    })

    return report


# ------------------------------------------
# SIDEBAR
# ------------------------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded successfully")

    sensitive = st.sidebar.selectbox("Sensitive Attribute Column", df.columns)
    target = st.sidebar.selectbox(
        "Target Column",
        [c for c in df.columns if c != sensitive]
    )

    detect = st.sidebar.button("🔍 Run Bias Analysis")

    if detect:
        result = bias_analysis(df, sensitive, target)

        # ------------------------------------------
        # WARNING IF TARGET WAS ENCODED
        # ------------------------------------------
        if result["target_encoded"]:
            st.warning(
                f"Target column `{target}` was non-numeric and has been "
                "automatically encoded for bias analysis."
            )

        # ------------------------------------------
        # DATASET OVERVIEW
        # ------------------------------------------
        st.header("Dataset Overview")
        c1, c2 = st.columns(2)
        c1.metric("Total Rows", len(df))
        c2.metric("Total Columns", len(df.columns))

        st.divider()

        # ------------------------------------------
        # BIAS STATUS
        # ------------------------------------------
        st.header("Bias Risk Assessment")

        if result["traffic"] == "green":
            st.success("LOW RISK: Dataset is largely balanced.")
        elif result["traffic"] == "yellow":
            st.warning("MODERATE RISK: Bias indicators detected.")
        else:
            st.error("HIGH RISK: Dataset correction required.")

        st.divider()

        # ------------------------------------------
        # VERY SMALL THUMBNAIL GRAPH
        # ------------------------------------------
        st.subheader("Bias Concentration (Thumbnail View)")

        col_thumb, _ = st.columns([1, 7])

        with col_thumb:
            fig_small, ax_small = plt.subplots(figsize=(1.2, 1.2))

            rep = result["rep"]
            ax_small.scatter(rep.index.astype(str), rep.values, s=12)
            ax_small.scatter(
                result["dominant_group"],
                result["dominant_ratio"],
                color="red",
                s=25
            )

            ax_small.set_ylim(0, 1)
            ax_small.set_xticks([])
            ax_small.set_yticks([])
            ax_small.set_title("Preview", fontsize=7)

            st.pyplot(fig_small, use_container_width=False)

        # ------------------------------------------
        # EXPANDABLE LARGE GRAPH
        # ------------------------------------------
        with st.expander("🔍 Expand for detailed visualization"):
            fig_big, ax_big = plt.subplots(figsize=(6, 5))

            ax_big.scatter(rep.index.astype(str), rep.values, s=100)
            ax_big.scatter(
                result["dominant_group"],
                result["dominant_ratio"],
                color="red",
                s=200,
                label="Over-represented group"
            )

            ax_big.set_ylim(0, 1)
            ax_big.set_ylabel("Proportion of Rows")
            ax_big.set_xlabel(sensitive)
            ax_big.legend()

            st.pyplot(fig_big)

        st.divider()

        # ------------------------------------------
        # TECHNICAL RECOMMENDATIONS
        # ------------------------------------------
        st.subheader("🔧 Technical Dataset Correction Guidelines")

        if result["rep_issue"]:
            affected_rows = int(result["dominant_ratio"] * len(df))

            st.warning(
                f"""
**Representation Bias Identified**

- **Column:** `{sensitive}`
- **Dominant category:** `{result['dominant_group']}`
- **Approx. affected rows:** {affected_rows} / {len(df)}

**Corrective Actions:**
1. Collect additional samples for minority categories.
2. Apply random undersampling on dominant rows.
3. Use stratified sampling during train–test split.
"""
            )
        else:
            st.success(f"No major representation imbalance detected in `{sensitive}`.")

        if result["label_issue"]:
            st.warning(
                f"""
**Label Bias Detected**

- **Target column:** `{target}`
- **Issue:** Outcome disparity across `{sensitive}` groups.

**Corrective Actions:**
1. Review labeling criteria.
2. Normalize decision rules.
3. Re-label affected samples.
"""
            )
        else:
            st.success(f"Target distribution in `{target}` is consistent.")

        if result["fairness_issue"]:
            st.warning(
                f"""
**Fairness Risk Observed**

- SPD: `{result['spd']:.2f}`
- DI: `{result['di']:.2f}`

**Corrective Actions:**
1. Avoid using `{sensitive}` as model input.
2. Balance dataset before training.
3. Validate fairness post-training.
"""
            )
        else:
            st.success("Fairness metrics are within acceptable limits.")

        st.divider()

        # ------------------------------------------
        # DETAILED TABLES
        # ------------------------------------------
        with st.expander("📘 Detailed Statistics"):
            st.write("Sensitive attribute distribution:")
            st.dataframe(result["rep"])

            st.write("Group-wise target averages:")
            st.dataframe(result["label_dist"])

else:
    st.info("Upload a CSV file from the sidebar to begin analysis.")

