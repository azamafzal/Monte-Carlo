
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Helper function
def compute_ci(power, n, z=1.96):
    se = np.sqrt((power * (1 - power)) / n)
    return power - z * se, power + z * se

# Simulation for t-test
def simulate_ttest(sample_size, effect_size, sd, alpha, num_simulations):
    p_vals = []
    for _ in range(num_simulations):
        g1 = np.random.normal(0, sd, sample_size)
        g2 = np.random.normal(effect_size, sd, sample_size)
        _, p = stats.ttest_ind(g1, g2)
        p_vals.append(p)
    power = np.mean(np.array(p_vals) < alpha)
    ci = compute_ci(power, num_simulations)
    return power, ci, p_vals

# Simulation for clustered logistic regression
def simulate_clustered_logistic(num_clusters, cluster_size, beta, icc, alpha, num_simulations):
    p_vals = []
    for _ in range(num_simulations):
        cluster_effects = np.random.normal(0, np.sqrt(icc), num_clusters)
        data = []
        for i in range(num_clusters):
            x = np.random.normal(0, 1, cluster_size)
            group_mean = cluster_effects[i]
            logits = beta * x + group_mean
            prob = 1 / (1 + np.exp(-logits))
            y = np.random.binomial(1, prob)
            cluster_df = pd.DataFrame({'x': x, 'cluster': i, 'y': y})
            data.append(cluster_df)
        df = pd.concat(data)
        X = sm.add_constant(df['x'])
        model = sm.Logit(df['y'], X).fit(disp=0)
        p_vals.append(model.pvalues[1])
    power = np.mean(np.array(p_vals) < alpha)
    ci = compute_ci(power, num_simulations)
    return power, ci, p_vals

def main():
    st.title("Monte Carlo Simulation Toolkit")

    test_type = st.sidebar.selectbox("Select Simulation Type", ["t-test", "Clustered Logistic Regression"])

    if test_type == "t-test":
        st.header("Two-sample t-test Simulation")
        sample_size = st.slider("Sample size per group", 10, 500, 30, 10)
        effect_size = st.slider("Effect size (Cohen's d)", 0.0, 2.0, 0.5, 0.1)
        sd = st.slider("Standard deviation", 0.1, 5.0, 1.0, 0.1)
        alpha = st.slider("Alpha (significance level)", 0.01, 0.10, 0.05, 0.01)
        sims = st.slider("Number of simulations", 100, 10000, 1000, 100)

        if st.button("Run Simulation"):
            power, ci, p_vals = simulate_ttest(sample_size, effect_size, sd, alpha, sims)
            st.success(f"Estimated Power: {power:.3f}")
            st.info(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

            fig, ax = plt.subplots()
            sns.histplot(p_vals, bins=30, kde=True, ax=ax)
            ax.axvline(x=alpha, color='red', linestyle='--', label=f'Alpha = {alpha}')
            ax.set_title("Distribution of p-values")
            ax.set_xlabel("p-value")
            ax.legend()
            st.pyplot(fig)

    if test_type == "Clustered Logistic Regression":
        st.header("Clustered Logistic Regression Simulation")
        num_clusters = st.slider("Number of clusters", 2, 50, 10)
        cluster_size = st.slider("Cluster size", 5, 200, 20)
        beta = st.slider("Beta coefficient", 0.0, 5.0, 0.5, 0.1)
        icc = st.slider("Intraclass correlation (ICC)", 0.0, 0.5, 0.05, 0.01)
        alpha = st.slider("Alpha (significance level)", 0.01, 0.10, 0.05, 0.01)
        sims = st.slider("Number of simulations", 100, 10000, 1000, 100)

        if st.button("Run Clustered Simulation"):
            power, ci, p_vals = simulate_clustered_logistic(num_clusters, cluster_size, beta, icc, alpha, sims)
            st.success(f"Estimated Power: {power:.3f}")
            st.info(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")

            fig, ax = plt.subplots()
            sns.histplot(p_vals, bins=30, kde=True, ax=ax)
            ax.axvline(x=alpha, color='red', linestyle='--', label=f'Alpha = {alpha}')
            ax.set_title("Distribution of p-values")
            ax.set_xlabel("p-value")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
