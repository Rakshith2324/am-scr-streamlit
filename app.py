import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from bayes_model import run_scenario

st.set_page_config(page_title="AM Supply Chain Resilience Simulator", layout="wide")

st.title("Additive Manufacturing Supply Chain Resilience Simulator")

st.markdown(
    """
This tool implements a Bayesian causal model linking:

- Supply Risk (SR) → Supply Chain Complexity (SCC)
- SCC and Sourcing Flexibility (SF) → AM Scalability (AMS)
- AMS and SR → Supply Chain Resilience (SCR)
"""
)

st.sidebar.header("Scenario Inputs (0–100)")

sr_val = st.sidebar.slider(
    "Supply Risk (SR)",
    min_value=0,
    max_value=100,
    value=60,
    help="0 = very secure upstream supply, 100 = extremely risky and disruption-prone.",
)

scc_val = st.sidebar.slider(
    "Supply Chain Complexity (SCC)",
    min_value=0,
    max_value=100,
    value=50,
    help="0 = very simple network and products, 100 = highly complex multi-tier BOMs.",
)

sf_val = st.sidebar.slider(
    "Sourcing Flexibility (SF)",
    min_value=0,
    max_value=100,
    value=40,
    help="0 = single-sourced, rigid, 100 = highly flexible multi-sourcing and backup capacity.",
)

st.sidebar.markdown("---")

draws = st.sidebar.slider("Posterior draws", 500, 4000, 2000, step=500)
tune = st.sidebar.slider("Tuning steps", 500, 4000, 1000, step=500)

if st.button("Run Simulation"):
    st.write("Sampling posterior… this may take a few seconds.")

    trace, ams_samples, scr_samples, summary = run_scenario(
        sr_val, scc_val, sf_val, draws=draws, tune=tune
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Posterior distribution of AM Scalability (AMS)")
        fig_ams, ax_ams = plt.subplots()
        ax_ams.hist(ams_samples, bins=40, color="skyblue", edgecolor="black", alpha=0.8)
        ax_ams.set_xlabel("AMS score (0–100)")
        ax_ams.set_ylabel("Frequency")
        ax_ams.axvline(summary["AMS_mean"], color="red", linestyle="--", label="Mean")
        ax_ams.legend()
        st.pyplot(fig_ams)

        st.markdown(
            f"""
**AMS summary:**

- Mean: {summary['AMS_mean']:.1f}  
- Std dev: {summary['AMS_std']:.1f}  
- 5th–95th percentile: [{summary['AMS_p5']:.1f}, {summary['AMS_p95']:.1f}]
"""
        )

    with col2:
        st.subheader("Posterior distribution of Supply Chain Resilience (SCR)")
        fig_scr, ax_scr = plt.subplots()
        ax_scr.hist(scr_samples, bins=40, color="lightgreen", edgecolor="black", alpha=0.8)
        ax_scr.set_xlabel("SCR score (0–100)")
        ax_scr.set_ylabel("Frequency")
        ax_scr.axvline(summary["SCR_mean"], color="red", linestyle="--", label="Mean")
        ax_scr.legend()
        st.pyplot(fig_scr)

        st.markdown(
            f"""
**SCR summary:**

- Mean: {summary['SCR_mean']:.1f}  
- Std dev: {summary['SCR_std']:.1f}  
- 5th–95th percentile: [{summary['SCR_p5']:.1f}, {summary['SCR_p95']:.1f}]
"""
        )

    st.markdown("---")
    st.subheader("Model diagnostics (optional)")

    with st.expander("Show ArviZ summary table"):
        st.write(az.summary(trace, var_names=["AMS_latent", "SCR_latent"]))
else:
    st.info("Set your scenario on the left and click Run Simulation to see AMS and SCR distributions.")
