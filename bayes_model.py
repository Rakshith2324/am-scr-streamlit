import numpy as np
import pymc as pm
import arviz as az


def _scale_0_100_to_0_1(x):
    """Utility: map 0–100 slider values to 0–1."""
    return np.asarray(x) / 100.0


def build_model(sr_val, scc_val, sf_val):
    """Build a PyMC model for one scenario."""

    sr = _scale_0_100_to_0_1(sr_val)
    scc_obs = _scale_0_100_to_0_1(scc_val)
    sf = _scale_0_100_to_0_1(sf_val)

    with pm.Model() as model:
        alpha_scc = pm.Normal("alpha_scc", mu=0.5, sigma=0.2)
        beta_scc_sr = pm.Normal("beta_scc_sr", mu=0.6, sigma=0.2)
        sigma_scc = pm.HalfNormal("sigma_scc", sigma=0.2)

        alpha_ams = pm.Normal("alpha_ams", mu=0.5, sigma=0.2)
        beta_ams_scc = pm.Normal("beta_ams_scc", mu=-0.5, sigma=0.3)
        beta_ams_sf = pm.Normal("beta_ams_sf", mu=0.5, sigma=0.3)
        beta_ams_inter = pm.Normal("beta_ams_inter", mu=0.4, sigma=0.3)
        sigma_ams = pm.HalfNormal("sigma_ams", sigma=0.2)

        alpha_scr = pm.Normal("alpha_scr", mu=0.5, sigma=0.2)
        beta_scr_ams = pm.Normal("beta_scr_ams", mu=0.8, sigma=0.2)
        beta_scr_sr = pm.Normal("beta_scr_sr", mu=-0.4, sigma=0.2)
        sigma_scr = pm.HalfNormal("sigma_scr", sigma=0.2)

        mu_scc = alpha_scc + beta_scc_sr * sr
        scc_latent = pm.Normal("SCC_latent", mu=mu_scc, sigma=sigma_scc)

        interaction = scc_latent * sf
        mu_ams = (
            alpha_ams
            + beta_ams_scc * scc_latent
            + beta_ams_sf * sf
            + beta_ams_inter * interaction
        )
        ams_latent = pm.Normal("AMS_latent", mu=mu_ams, sigma=sigma_ams)

        mu_scr = alpha_scr + beta_scr_ams * ams_latent + beta_scr_sr * sr
        scr_latent = pm.Normal("SCR_latent", mu=mu_scr, sigma=sigma_scr)

        pm.Normal("SCC_obs", mu=scc_latent, sigma=0.1, observed=scc_obs)

    return model


def run_scenario(sr_val, scc_val, sf_val, draws=2000, tune=1000, chains=2, cores=2):
    """Build and sample the Bayesian model for a single scenario."""

    model = build_model(sr_val, scc_val, sf_val)

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=False,
        )

    ams_samples = trace.posterior["AMS_latent"].values.flatten()
    scr_samples = trace.posterior["SCR_latent"].values.flatten()

    ams_100 = np.clip(ams_samples * 100.0, 0.0, 100.0)
    scr_100 = np.clip(scr_samples * 100.0, 0.0, 100.0)

    summary = {
        "AMS_mean": float(np.mean(ams_100)),
        "AMS_std": float(np.std(ams_100)),
        "AMS_p5": float(np.percentile(ams_100, 5)),
        "AMS_p95": float(np.percentile(ams_100, 95)),
        "SCR_mean": float(np.mean(scr_100)),
        "SCR_std": float(np.std(scr_100)),
        "SCR_p5": float(np.percentile(scr_100, 5)),
        "SCR_p95": float(np.percentile(scr_100, 95)),
    }

    return trace, ams_100, scr_100, summary
