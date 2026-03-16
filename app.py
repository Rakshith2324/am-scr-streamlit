import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from bayes_model import run_scenario


# ---------- Helper functions ----------

def classify_level(score):
    """Return 'Low', 'Medium', or 'High' for a 0–100 score."""
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"


def insight_text(sr, scc, sf, ams_mean, scr_mean):
    """Generate a few simple bullet-point insights."""
    msgs = []

    # Overall levels
    msgs.append(
        f"Overall AM scalability is **{classify_level(ams_mean)}** "
        f"({ams_mean:.0f}/100) and resilience is **{classify_level(scr_mean)}** "
        f"({scr_mean:.0f}/100)."
    )

    # Drivers based on inputs
    if sr >= 70:
        msgs.append(
            "High upstream **supply risk** is dragging resilience down; "
            "consider multi-sourcing or diversifying raw material suppliers."
        )
    elif sr <= 30:
        msgs.append(
            "Upstream supply risk is relatively **low**, which supports higher resilience."
        )

    if scc >= 70:
        msgs.append(
            "High **supply chain complexity** makes it harder to scale AM; "
            "simplifying BOMs or consolidating parts can help."
        )
    elif scc <= 30:
        msgs.append(
            "A relatively **simple network and BOM** structure makes AM scaling easier."
        )

    if sf >= 70:
        msgs.append(
            "Strong **sourcing flexibility** (multi-sourcing / backup capacity) "
            "is supporting both AM scalability and resilience."
        )
    elif sf <= 30:
        msgs.append(
            "Limited **sourcing flexibility** is a bottleneck; adding backup capacity "
            "or alternative suppliers could significantly improve resilience."
        )

    return msgs


# ---------- Streamlit layout ----------

st.set_page_config(
    page_title="AM Supply Chain Resilience Simulator",
    layout="wide",
)

st.title("Additive Manufacturing Supply Chain Resilience Simulator")

st.markdown(
    """
This tool uses a research-based causal model to estimate:

- **AM Scalability (AMS)** – how easily your additive manufacturing setup can scale.
- **Supply Chain Resilience (SCR)** – how well your supply chain can absorb and recover from disruptions.

You choose the scenario (risk, complexity, flexibility); the model simulates many plausible futures
and summarizes the expected scores and their likely range.[web:12]
"""
)

# ---- Sidebar: simple scenario inputs ----
st.sidebar.header("Scenario Inputs (0–100)")

sr_val = st.sidebar.slider(
    "Upstream Supply Risk",
    min_value=0,
    max_value=100,
    value=50,
    help="0 = very secure, diversified supply; 100 = highly risky, disruption-prone supply.",
)

scc_val = st.sidebar.slider(
    "Supply Chain Complexity",
    min_value=0,
    max_value=100,
    value=50,
    help="0 = very simple network and BOM; 100 = many tiers and complex BOMs.",
)

sf_val = st.sidebar.slider(
    "Sourcing Flexibility",
    min_value=0,
    max_value=100,
    value=50,
    help="0 = single-sourced, rigid; 100 = multi-sourced with backup capacity and flexible contracts.",
)

# Optional extra: AM maturity as a quick preset that nudges SF
st.sidebar.markdown("---")
am_maturity = st.sidebar.selectbox(
    "AM Maturity Level (optional)",
    ["Not using AM", "Pilot", "Scaling", "Mature"],
    help="This adjusts sourcing flexibility slightly behind the scenes.",
)

# Adjust SF slightly based on maturity (small effect so it doesn't dominate)
if am_maturity == "Not using AM":
    sf_effect = -10
elif am_maturity == "Pilot":
    sf_effect = 0
elif am_maturity == "Scaling":
    sf_effect = 5
else:  # Mature
    sf_effect = 10

effective_sf = int(np.clip(sf_val + sf_effect, 0, 100))

st.sidebar.markdown(f"**Effective sourcing flexibility used in model:** {effective_sf}/100")

# Fixed sampling settings (no sliders needed)
DRAWS = 2000
TUNE = 1000

if st.button("Run Simulation"):
    with st.spinner("Running simulations for this scenario…"):
        trace, ams_samples, scr_samples, summary = run_scenario(
            sr_val, scc_val, effective_sf, draws=DRAWS, tune=TUNE
        )

    ams_mean = summary["AMS_mean"]
    scr_mean = summary["SCR_mean"]

    # ---------- High-level cards ----------
    col_top1, col_top2 = st.columns(2)

    with col_top1:
        st.subheader("AM Scalability (AMS)")
        st.markdown(
            f"""
**Score:** {ams_mean:.1f} / 100  
**Level:** **{classify_level(ams_mean)}**
"""
        )

    with col_top2:
        st.subheader("Supply Chain Resilience (SCR)")
        st.markdown(
            f"""
**Score:** {scr_mean:.1f} / 100  
**Level:** **{classify_level(scr_mean)}**
"""
        )

    # ---------- Insights ----------
    st.markdown("---")
    st.subheader("Key Insights for this Scenario")

    for msg in insight_text(sr_val, scc_val, effective_sf, ams_mean, scr_mean):
        st.markdown(f"- {msg}")

    st.markdown(
        """
The scores are based on a Bayesian model that encodes how supply risk, complexity,
and sourcing flexibility influence AM scalability and resilience.[web:12]
"""
    )

    # ---------- Advanced: show uncertainty details ----------
    st.markdown("---")
    with st.expander("Advanced: see full range of possible scores and diagnostics"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Range of possible AM Scalability scores (AMS)**")
            fig_ams, ax_ams = plt.subplots()
            ax_ams.hist(
                ams_samples,
                bins=40,
                color="skyblue",
                edgecolor="black",
                alpha=0.8,
            )
            ax_ams.set_xlabel("AMS score (0–100)")
            ax_ams.set_ylabel("Count")
            ax_ams.axvline(ams_mean, color="red", linestyle="--", label="Mean")
            ax_ams.legend()
            st.pyplot(fig_ams)

            st.markdown(
                f"""
- Expected value (mean): **{ams_mean:.1f}**  
- Typical range (5th–95th percentile): **{summary['AMS_p5']:.1f} – {summary['AMS_p95']:.1f}**
"""
            )

        with col2:
            st.markdown("**Range of possible Resilience scores (SCR)**")
            fig_scr, ax_scr = plt.subplots()
            ax_scr.hist(
                scr_samples,
                bins=40,
                color="lightgreen",
                edgecolor="black",
                alpha=0.8,
            )
            ax_scr.set_xlabel("SCR score (0–100)")
            ax_scr.set_ylabel("Count")
            ax_scr.axvline(scr_mean, color="red", linestyle="--", label="Mean")
            ax_scr.legend()
            st.pyplot(fig_scr)

            st.markdown(
                f"""
- Expected value (mean): **{scr_mean:.1f}**  
- Typical range (5th–95th percentile): **{summary['SCR_p5']:.1f} – {summary['SCR_p95']:.1f}**
"""
            )

        st.markdown("**Sampling diagnostics (for technical readers)**")
        st.write(az.summary(trace, var_names=["AMS_latent", "SCR_latent"]))

else:
    st.info("Choose your scenario on the left and click **Run Simulation** to get results.")
