import sys
import os
import numpy as np
import pandas as pd

# Add current workspace directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import get_bvl_data
from econometrics import get_pen_prices_and_volumes, filter_assets_by_volume, fit_garch_models
from optimization import simulate_garch_paths, optimize_portfolio, calculate_cvar

def main():
    print("==============================================")
    print("VERIFICATION SCRIPT: PORTFOLIO RISK PIPELINE")
    print("==============================================\n")
    
    # 1. Verification of Data Pipeline
    print("[Verification 1] Executing data download and cleaning...")
    df = get_bvl_data()
    print("Data download success.")
    print(f"Shape: {df.shape}")
    print("Descriptive stats for Volume (verifying minimum volume is 0.0):")
    desc_vol = df["Volume"].describe()
    print(desc_vol.loc[["min", "mean", "max"]])
    
    # Verify that all minimum volumes are indeed 0.0 (or at least one is 0.0 and close to it for others if they never miss trading, but most BVL stocks miss trading)
    zero_vols_check = (df["Volume"] == 0).sum()
    print("\nNumber of days with 0 volume per asset (verifying ffill + 0 vol logic):")
    print(zero_vols_check)
    
    # 2. Verification of Econometrics Layer (GARCH p-values)
    print("\n[Verification 2] Running currency standardization and asset filtering...")
    prices_pen, volume_pen = get_pen_prices_and_volumes(df)
    prices_filtered, _, valid_tickers = filter_assets_by_volume(prices_pen, volume_pen)
    
    print("\nRunning GARCH(1,1) fits and checking p-values...")
    _, std_residuals, garch_models, _ = fit_garch_models(prices_filtered)
    
    # Let's print out coefficient p-values to verify significance
    significance_table = []
    for t in valid_tickers:
        res = garch_models[t]
        alpha_pval = res.pvalues.get("alpha[1]", np.nan)
        beta_pval = res.pvalues.get("beta[1]", np.nan)
        significance_table.append({
            "Asset": t,
            "ARCH p-val (alpha)": alpha_pval,
            "GARCH p-val (beta)": beta_pval,
            "Beta Significant (5%)": "Yes" if beta_pval < 0.05 else "No"
        })
    sig_df = pd.DataFrame(significance_table)
    print("\nCoefficient Significance Summary:")
    print(sig_df.to_string(index=False))
    
    # 3. Verification of Optimization (constraints check)
    print("\n[Verification 3] Simulating GARCH Bootstrap paths and optimizing portfolio...")
    np.random.seed(42)
    sim_rets = simulate_garch_paths(garch_models, std_residuals, n_sims=10000, n_days=20)
    w_opt, opt_res = optimize_portfolio(sim_rets, valid_tickers, 0.95)
    
    print("\nVerification of Constraints:")
    sum_weights = np.sum(w_opt)
    max_weight = np.max(w_opt)
    min_weight = np.min(w_opt)
    
    print(f"1. Sum of weights (should be 1.0): {sum_weights:.8f}")
    print(f"2. Minimum weight (should be >= 0.0): {min_weight:.8f}")
    print(f"3. Maximum weight (should be <= 0.15): {max_weight:.8f}")
    
    # Assert checks
    assert np.isclose(sum_weights, 1.0, atol=1e-5), "Constraint failure: Sum of weights is not 1.0"
    assert min_weight >= -1e-6, "Constraint failure: Short-selling detected (weight < 0)"
    assert max_weight <= 0.15 + 1e-6, "Constraint failure: Diversification ceiling exceeded (weight > 15%)"
    
    print("\nAll constraints verified successfully!")
    print(f"CVaR at 95% for capital S/. 3,200: S/. {calculate_cvar(w_opt, sim_rets, 0.95)*3200:.2f}")
    print("==============================================\n")

if __name__ == "__main__":
    main()
