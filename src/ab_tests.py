import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
import math
import itertools
import re
import json
import importlib

import matplotlib.pyplot as plt
import seaborn as sns

import pingouin as pg
import scikit_posthocs as sp

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, chi2_contingency, fisher_exact, norm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from src.framework import extract_p_value 

def configure_ab_test(metric_type, is_equal_variance, is_normal_distribution, num_variants, variant_ratio, sample_sizes, is_strong_outlier_effect, size_threshold = 30):
    print(f'metric_type: {metric_type}')
    print(f'is_normal_distribution: {is_normal_distribution}')
    print(f'is_equal_variance: {is_equal_variance}')
    print(f'num_variants: {num_variants}')
    print(f'sample_sizes: {sample_sizes}')
    print(f'variant_ratio: {np.round(variant_ratio, 2)}')
    print(f'is_strong_outlier_effect: {is_strong_outlier_effect}')
    
    ab_test_config = {
        # Bi variant tests
        "use_mann_whitney_u_test": False, # continuous, not normal, works with unbalanced samples
        "use_welchs_t_test": False, # continuous, normal, different variances, works with unbalanced samples (if low unbalance applied although same variants)
        "use_t_test": False, # continuous, normal, equal variances, not recommended if unbalanced
        "use_fisher_exact_test": False, # proportions, small sample size, works with unbalanced samples, # for proportions, only size matters due to central limit theorem
        "use_two_proportion_z_test": False, # proportions, large sample size, works with unbalanced samples, used alone or post Pearson Chi-square with Bonferroni for multiple variants, # for proportions, only size matters due to central limit theorem
        ## Multiple variants tests
        "use_anova_test": False, # continuous, normal, not recommended if unbalanced, equal variance
        "use_welch_anova_test": False, # continuous, normal, if unbalanced, different variance
        "use_kruskal_wallis_test": False, # continuous, not normal, works with unbalanced samples
        "use_pearson_chi_square_test": False, # proportions, works with unbalanced samples but needs correction
        ## Multiple variants post-pairs tests
        "use_tukey_hsd_test": False, # post anova, continuous, not recommended if unbalanced
        "use_games_howell_test": False, # post anova, continuous, for unbalanced samples
        "use_dunn_test": False, # post kruskal wallis, continuous, needs bonferroni, works with unbalanced samples
        ## Multiple variants correction
        "use_bonferroni_correction": False, # if more than 2 variants and tukey pr games howell is not used
        ## Unbalance data: N = A/B, if N < 2.5 use normal testing, if N < 5x use balance resampling, if N > 5x use bootstraping
        "use_balance_resampling": False, # for unbalanced sample sizes, equalizing groups, downsampling,
        "use_bootstraping": False, # for small sample sizes or unbalanced groups, estimating confidence intervals, upsampling,
        ## Additional techniques
        "use_bayesian_test": False, # for probabilistic interpretation, alternative to p-values, small samples
        "use_permutation_test": False # for distribution-free significance testing, alternative to t-tests or z-tests, extrange distributions
    }
    
    # Unbalance techniques
    if variant_ratio >= 2.5 and variant_ratio < 5:
        ab_test_config["use_balance_resampling"] = True
    elif variant_ratio >= 5:
        ab_test_config["use_bootstraping"] = True

    if metric_type == 'continuous':
        # Bi variant tests
        if num_variants == 2:
            if not is_normal_distribution:
                ab_test_config["use_mann_whitney_u_test"] = True
            else:
                if is_equal_variance:
                    if variant_ratio < 1.5:
                        ab_test_config["use_t_test"] = True 
                    elif 1.5 <= variant_ratio <= 2.5:
                        ab_test_config["use_welchs_t_test"] = True 
                    else:  
                        ab_test_config["use_t_test"] = True
                else:
                    ab_test_config["use_welchs_t_test"] = True 
        # Multiple variant tests
        elif num_variants > 2:
            if is_normal_distribution:
                if is_equal_variance:
                    if variant_ratio < 1.5:
                        ab_test_config["use_anova_test"] = True
                        ab_test_config["use_tukey_hsd_test"] = True
                    elif 1.5 <= variant_ratio <= 2.5:
                        ab_test_config["use_welch_anova_test"] = True
                        ab_test_config["use_games_howell_test"] = True
                    else:  # variant_ratio > 2.5
                        ab_test_config["use_anova_test"] = True
                        ab_test_config["use_tukey_hsd_test"] = True 
                else:
                    ab_test_config["use_welch_anova_test"] = True
                    ab_test_config["use_games_howell_test"] = True  
            else:
                ab_test_config["use_kruskal_wallis_test"] = True
                ab_test_config["use_dunn_test"] = True
            
            if not (ab_test_config["use_tukey_hsd_test"] or 
                    ab_test_config["use_games_howell_test"]) or ab_test_config["use_dunn_test"]:
                ab_test_config["use_bonferroni_correction"] = True

    # Proportions metric
    elif metric_type == 'proportion':
        # Bi variant tests
        if num_variants == 2:
            if sample_sizes.min() < size_threshold:
                ab_test_config["use_fisher_exact_test"] = True
            else:
                ab_test_config["use_two_proportion_z_test"] = True
        # Multiple variant tests
        elif num_variants > 2:
            ab_test_config["use_pearson_chi_square_test"] = True
            ab_test_config["use_two_proportion_z_test"] = True
            ab_test_config["use_bonferroni_correction"] = True

    # Additional techniques for small sample sizes (threshold < size_threshold)
    if sample_sizes.min() < 1000 or not is_normal_distribution:
        ab_test_config["use_bayesian_test"] = True
        
    if sample_sizes.min() < 100 or (is_strong_outlier_effect and sample_sizes.min() < 2000):
        ab_test_config["use_permutation_test"] = True
    return ab_test_config

def apply_balance_resampling(selected_df, variant_column):
    min_count = selected_df[variant_column].value_counts().min()
    groups = [
        group_df.sample(n=min_count, random_state=42, replace=False)
        for _, group_df in selected_df.groupby(variant_column)
    ]
    return pd.concat(groups).reset_index(drop=True)

def apply_bootstraping(selected_df, variant_column):
    max_count = selected_df[variant_column].value_counts().max()
    groups = [
        group_df.sample(n=max_count, random_state=42, replace=True)
        for _, group_df in selected_df.groupby(variant_column)
    ]
    return pd.concat(groups).reset_index(drop=True)

def resample_data(selected_df, ab_test_config, variant_column):
    if ab_test_config.get("use_balance_resampling", False):
        print("use_balance_resampling")
        return apply_balance_resampling(selected_df, variant_column)
    elif ab_test_config.get("use_bootstraping", False):
        print("apply_bootstraping")
        return apply_bootstraping(selected_df, variant_column)
    else:
        return selected_df

def run_t_test(df, variant_column, metric_column):
    variants = df[variant_column].unique()
    groupA = df[df[variant_column] == variants[0]][metric_column]
    groupB = df[df[variant_column] == variants[1]][metric_column]
    stat, p_value = ttest_ind(groupA, groupB, equal_var=True)
    return {"stat": stat, "p_value": p_value}

def run_welchs_t_test(df, variant_column, metric_column):
    variants = df[variant_column].unique()
    groupA = df[df[variant_column] == variants[0]][metric_column]
    groupB = df[df[variant_column] == variants[1]][metric_column]
    stat, p_value = ttest_ind(groupA, groupB, equal_var=False)
    return {"stat": stat, "p_value": p_value}

def run_mann_whitney(df, variant_column, metric_column):
    variants = df[variant_column].unique()
    groupA = df[df[variant_column] == variants[0]][metric_column]
    groupB = df[df[variant_column] == variants[1]][metric_column]
    stat, p_value = mannwhitneyu(groupA, groupB, alternative='two-sided')
    return {"stat": stat, "p_value": p_value}

def run_two_proportion_z_test(df, variant_column, metric_column):
    agg = df.groupby(variant_column)[metric_column].agg(['sum', 'count']).reset_index()
    agg = agg.rename(columns={'sum': 'success_count', 'count': 'total_count'})
    
    results = []
    # iterate over all unique pairs of variants
    for group1, group2 in itertools.combinations(agg[variant_column].unique(), 2):
        row1 = agg.loc[agg[variant_column] == group1].iloc[0]
        row2 = agg.loc[agg[variant_column] == group2].iloc[0]
        
        success_counts = np.array([row1['success_count'], row2['success_count']])
        total_counts = np.array([row1['total_count'], row2['total_count']])
        
        stat, p_value = proportions_ztest(success_counts, total_counts)
        # compute difference in proportions
        prop1 = row1['success_count'] / row1['total_count']
        prop2 = row2['success_count'] / row2['total_count']
        diff = prop1 - prop2
        
        results.append({
            'group1': group1,
            'group2': group2,
            'diff': diff,
            'p_value': p_value,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        })
    
    return pd.DataFrame(results)

def run_fisher_exact_test(df, variant_column, metric_column):
    contingency_table = pd.crosstab(df[variant_column], df[metric_column])
    stat, p_value = fisher_exact(contingency_table)
    return {"stat": stat, "p_value": p_value}

def run_anova(df, variant_column, metric_column):
    groups = [df[df[variant_column] == variant][metric_column] for variant in df[variant_column].unique()]
    stat, p_value = f_oneway(*groups)
    return {"stat": stat, "p_value": p_value}

def run_welch_anova(df, variant_column, metric_column):
    res = pg.welch_anova(dv=metric_column, between=variant_column, data=df)
    f_stat = res["F"].iloc[0]
    p_value = res["p-unc"].iloc[0]
    return {"stat": f_stat, "p_value": p_value}

def run_kruskal_wallis(df, variant_column, metric_column):
    groups = [df[df[variant_column] == variant][metric_column] for variant in df[variant_column].unique()]
    stat, p_value = kruskal(*groups)
    return {"stat": stat, "p_value": p_value}

def run_chi_square(df, variant_column, metric_column):
    contingency_table = pd.crosstab(df[variant_column], df[metric_column])
    stat, p_value, _, _ = chi2_contingency(contingency_table)
    return {"stat": stat, "p_value": p_value}

def run_tukey_hsd(df, variant_column, metric_column):
    tukey = pairwise_tukeyhsd(endog=df[metric_column], groups=df[variant_column], alpha=0.05)
    table_data = tukey._results_table.data
    header = table_data[0]
    data_rows = table_data[1:]
    df_tukey = pd.DataFrame(data_rows, columns=header)
    df_tukey = df_tukey.rename(columns={
        'group1': 'group1',
        'group2': 'group2',
        'meandiff': 'diff',
        'p-adj': 'p_value',
        'lower': 'ci_lower',
        'upper': 'ci_upper'
    })
    return df_tukey[['group1', 'group2', 'diff', 'p_value', 'ci_lower', 'ci_upper']]

def run_games_howell(df, variant_column, metric_column):
    res = pg.pairwise_gameshowell(data=df, dv=metric_column, between=variant_column)
    res_standard = res.rename(columns={'A': 'group1', 'B': 'group2', 'diff': 'diff', 'pval': 'p_value'})
    res_standard['ci_lower'] = np.nan
    res_standard['ci_upper'] = np.nan
    return res_standard[['group1', 'group2', 'diff', 'p_value', 'ci_lower', 'ci_upper']]

def run_dunn_test(df, variant_column, metric_column):
    res = sp.posthoc_dunn(df, val_col=metric_column, group_col=variant_column, p_adjust='bonferroni')
    tidy_rows = []
    groups = res.index.tolist()
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if j > i:
                tidy_rows.append({'group1': g1, 'group2': g2, 'p_value': res.loc[g1, g2]})
    df_tidy = pd.DataFrame(tidy_rows)
    df_tidy['diff'] = np.nan
    df_tidy['ci_lower'] = np.nan
    df_tidy['ci_upper'] = np.nan
    return df_tidy[['group1', 'group2', 'diff', 'p_value', 'ci_lower', 'ci_upper']]

def apply_bonferroni_correction(tidy_df):
    p_values = pd.to_numeric(tidy_df['p_value'], errors='coerce').values
    mask = ~np.isnan(p_values)
    adjusted_p_values = multipletests(p_values[mask], method='bonferroni')[1]
    new_p_values = p_values.copy()
    new_p_values[mask] = adjusted_p_values
    tidy_df = tidy_df.copy()
    tidy_df['p_value'] = new_p_values
    return tidy_df

def run_ab_tests(test_config, df, variant_column, metric_column):
    results = {}

    # Continuous
    if test_config.get("use_t_test", False):
        results["t_test"] = run_t_test(df, variant_column, metric_column)
    if test_config.get("use_welchs_t_test", False):
        results["welchs_t_test"] = run_welchs_t_test(df, variant_column, metric_column)
    if test_config.get("use_mann_whitney_u_test", False):
        results["mann_whitney_u_test"] = run_mann_whitney(df, variant_column, metric_column)

    # Proportions
    if test_config.get("use_fisher_exact_test", False):
        results["fisher_exact_test"] = run_fisher_exact_test(df, variant_column, metric_column)
    if test_config.get("use_two_proportion_z_test", False):
        results["two_proportion_z_test"] = run_two_proportion_z_test(df, variant_column, metric_column)

    # Continuous - Multiple Variants
    if test_config.get("use_anova_test", False):
        results["anova_test"] = run_anova(df, variant_column, metric_column)
    if test_config.get("use_welch_anova_test", False):
        results["welch_anova_test"] = run_welch_anova(df, variant_column, metric_column)
    if test_config.get("use_kruskal_wallis_test", False):
        results["kruskal_wallis_test"] = run_kruskal_wallis(df, variant_column, metric_column)

    # Proportions - Multiple Variants
    if test_config.get("use_pearson_chi_square_test", False):
        results["pearson_chi_square_test"] = run_chi_square(df, variant_column, metric_column)

    return results

def run_post_hoc_tests(test_config, df, variant_column, metric_column):
    results = {}

    if test_config.get("use_tukey_hsd_test", False):
        results["tukey_hsd_test"] = run_tukey_hsd(df, variant_column, metric_column)
        
    if test_config.get("use_games_howell_test", False):
        results["games_howell_test"] = run_games_howell(df, variant_column, metric_column)

    if test_config.get("use_dunn_test", False):
        dunn_results = run_dunn_test(df, variant_column, metric_column)
        results["dunn_test"] = dunn_results
        if test_config.get("use_bonferroni_correction", False):
            results["bonferroni_correction"] = apply_bonferroni_correction(dunn_results)

    if test_config.get("use_pearson_chi_square_test", False):
        z_test_results = run_two_proportion_z_test(df, variant_column, metric_column)
        results["two_proportion_z_test"] = z_test_results
        if test_config.get("use_bonferroni_correction", False):
            results["bonferroni_correction"] = apply_bonferroni_correction(z_test_results)

    return results

def run_complete_ab_test(ab_test_config, selected_df, variant_column, metric_column, num_variants, alpha=0.05):
    ab_test_results = run_ab_tests(ab_test_config, selected_df, variant_column, metric_column)
    if num_variants > 2:
        p_value = extract_p_value(ab_test_results)
        if p_value is not None and p_value < alpha:
            post_hoc_results = run_post_hoc_tests(ab_test_config, selected_df, variant_column, metric_column)
            ab_test_results["post_hoc"] = post_hoc_results
    standardized_results = {}
    if "post_hoc" in ab_test_results and ab_test_results["post_hoc"]:
        standardized_results["post_hoc"] = {}
        for test_name, test_data in ab_test_results["post_hoc"].items():
            if hasattr(test_data, "to_dict"):
                comparisons = test_data.to_dict("records")
            elif isinstance(test_data, list):
                comparisons = test_data
            else:
                comparisons = []
            standardized_results["post_hoc"][test_name] = []
            for comp in comparisons:
                p_val = comp.get("p_value")
                standardized_results["post_hoc"][test_name].append({
                    "group1": comp.get("group1"),
                    "group2": comp.get("group2"),
                    "p_value": p_val,
                    "significant": p_val < alpha if p_val is not None else None
                })
    else:
        primary_test_key = next((k for k in ab_test_results.keys() if k != "post_hoc"), None)
        if primary_test_key is not None:
            primary_test_result = ab_test_results[primary_test_key]
            p_val = primary_test_result.get("p_value")
            group1 = primary_test_result.get("group1")
            group2 = primary_test_result.get("group2")
            if group1 is None or group2 is None:
                groups = selected_df[variant_column].unique()
                if len(groups) >= 2:
                    group1, group2 = groups[0], groups[1]
            standardized_results["overall_test"] = {
                "test": primary_test_key,
                "group1": group1,
                "group2": group2,
                "p_value": p_val,
                "significant": p_val < alpha if p_val is not None else None
            }
    rows = []
    if "overall_test" in standardized_results:
        overall = standardized_results["overall_test"]
        rows.append({
            "test": overall.get("test"),
            "group1": overall.get("group1"),
            "group2": overall.get("group2"),
            "p_value": overall.get("p_value"),
            "significant": overall.get("significant")
        })
    if "post_hoc" in standardized_results:
        for test_name, comparisons in standardized_results["post_hoc"].items():
            for comp in comparisons:
                rows.append({
                    "test": test_name,
                    "group1": comp.get("group1"),
                    "group2": comp.get("group2"),
                    "p_value": comp.get("p_value"),
                    "significant": comp.get("significant")
                })
    df = pd.DataFrame(rows)
    if num_variants > 2:
        allowed_tests = ["bonferroni_correction", "tukey_hsd_test", "games_howell_test"]
        df = df[df["test"].isin(allowed_tests)]
    df = df.drop(columns={"test"})
    return df

def bayesian_ab_test(control_data, test_data, num_samples=10000):
    control_mean = np.mean(control_data)
    control_std = np.std(control_data, ddof=1)
    test_mean = np.mean(test_data)
    test_std = np.std(test_data, ddof=1)
    control_samples = np.random.normal(control_mean, control_std, num_samples)
    test_samples = np.random.normal(test_mean, test_std, num_samples)
    prob_test_better = np.mean(test_samples > control_samples)
    return {"bayesian_mean_difference": test_mean - control_mean, "bayesian_probability": prob_test_better}

def permutation_test(control_data, test_data, num_permutations=10000):
    observed_diff = np.mean(test_data) - np.mean(control_data)
    combined = np.concatenate([control_data, test_data])
    count_greater = 0
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_control = combined[:len(control_data)]
        new_test = combined[len(control_data):]
        permuted_diff = np.mean(new_test) - np.mean(new_control)
        if abs(permuted_diff) >= abs(observed_diff):
            count_greater += 1
    p_value = count_greater / num_permutations
    return {"permutation_observed_difference": observed_diff, "permutation_p_value": p_value}

def apply_additional_tests(ab_test_config, selected_df, variant_column, metric_column):
    results = {}
    groups = {}
    for variant, df_variant in selected_df.groupby(variant_column):
        groups[variant] = df_variant[metric_column].values
    variants = list(groups.keys())
    pairwise_tests = {}
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            pair = f"{variants[i]} vs {variants[j]}"
            pairwise_tests[pair] = {}
            if ab_test_config.get("use_bayesian_test", False):
                pairwise_tests[pair].update(bayesian_ab_test(groups[variants[i]], groups[variants[j]]))
            if ab_test_config.get("use_permutation_test", False):
                pairwise_tests[pair].update(permutation_test(groups[variants[i]], groups[variants[j]]))
    results["pairwise_tests"] = pairwise_tests
    return results