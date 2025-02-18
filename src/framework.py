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

def print_title(title, line_length = 60, symbol = '-'):
    separator = symbol * ((line_length - len(title) - 2) // 2)
    print(f"{separator} {title} {separator}")

def format_columns(df, datetime_columns=[], int64_columns=[], float64_columns=[], str_columns=[]):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in int64_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')  

    for col in float64_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')

    for col in str_columns:
        if col in df.columns:
            df[col] = df[col].astype('str')
    
    return df

def unbalance_dataset(selected_df, variant_column, imbalance_factor):
    counts = selected_df[variant_column].value_counts()
    max_count = counts.max()
    max_variants = counts[counts == max_count].index.tolist()
    desired_count = int(max_count / imbalance_factor)
    frames = []
    for variant, _ in counts.items():
        df_variant = selected_df[selected_df[variant_column] == variant]
        if variant in max_variants:
            frames.append(df_variant)
        else:
            if len(df_variant) < desired_count:
                df_variant = df_variant.sample(n=desired_count, random_state=42, replace=True)
            else:
                df_variant = df_variant.sample(n=desired_count, random_state=42, replace=False)
            frames.append(df_variant)
    return pd.concat(frames).reset_index(drop=True)

def extract_p_value(json_data):
    first_key = next(iter(json_data))
    p_value = json_data[first_key].get("p_value")
    return p_value

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

def plot_distributions(selected_df, variant_column, metric_column, alpha=0.05):
    variants = selected_df[variant_column].unique()
    z = norm.ppf(1 - alpha/2)  # Critical value based on alpha
    
    if len(variants) == 2:
        plt.figure(figsize=(16, 4))
        for variant in variants:
            subset = selected_df[selected_df[variant_column] == variant]
            mean_val = subset[metric_column].mean()
            std_val = subset[metric_column].std()
            n = len(subset)
            se = std_val / math.sqrt(n)
            lower_ci = mean_val - z * se
            upper_ci = mean_val + z * se
            x_min = lower_ci - 0.2 * (upper_ci - lower_ci)
            x_max = upper_ci + 0.2 * (upper_ci - lower_ci)
            x_values = np.linspace(x_min, x_max, 200)
            pdf = norm.pdf(x_values, mean_val, se)
            plt.plot(x_values, pdf, lw=2, label=variant)
            plt.axvline(mean_val, color='green', linestyle='-',alpha=0.7)
            plt.axvline(lower_ci, color='red', linestyle='--', alpha=0.5)
            plt.axvline(upper_ci, color='red', linestyle='--', alpha=0.5)
            plt.fill_between(x_values, pdf, where=(x_values >= lower_ci) & (x_values <= upper_ci),
                             color='red', alpha=0.2)
            pdf_mean = norm.pdf(mean_val, mean_val, se)
            plt.text(mean_val, pdf_mean * 1.05, f"{mean_val:.2f}",
                     ha='center', va='bottom', color='green', fontsize=10)
            pdf_lower = norm.pdf(lower_ci, mean_val, se)
            pdf_upper = norm.pdf(upper_ci, mean_val, se)
            plt.text(lower_ci, pdf_lower * 1.05, f"Lower CI: {lower_ci:.2f}",
                     ha='center', va='bottom', color='red', fontsize=10)
            plt.text(upper_ci, pdf_upper * 1.05, f"Upper CI: {upper_ci:.2f}",
                     ha='center', va='bottom', color='red', fontsize=10)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.legend(title="Variant", fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.show()
        
    else:
        pairs = list(itertools.combinations(variants, 2))
        n_pairs = len(pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 4 * n_pairs))
        if n_pairs == 1:
            axes = [axes]
        for idx, pair in enumerate(pairs):
            ax = axes[idx]
            for variant in pair:
                subset = selected_df[selected_df[variant_column] == variant]
                mean_val = subset[metric_column].mean()
                std_val = subset[metric_column].std()
                n = len(subset)
                se = std_val / math.sqrt(n)
                lower_ci = mean_val - z * se
                upper_ci = mean_val + z * se
                x_min = lower_ci - 0.2 * (upper_ci - lower_ci)
                x_max = upper_ci + 0.2 * (upper_ci - lower_ci)
                x_values = np.linspace(x_min, x_max, 200)
                pdf = norm.pdf(x_values, mean_val, se)
                ax.plot(x_values, pdf, lw=2, label=variant)
                ax.axvline(mean_val, color='green', linestyle='-', alpha=0.7)
                ax.axvline(lower_ci, color='red', linestyle='--', alpha=0.5)
                ax.axvline(upper_ci, color='red', linestyle='--', alpha=0.5)
                ax.fill_between(x_values, pdf, where=(x_values >= lower_ci) & (x_values <= upper_ci),
                                color='red', alpha=0.2)
                pdf_mean = norm.pdf(mean_val, mean_val, se)
                ax.text(mean_val, pdf_mean * 1.05, f"{mean_val:.2f}",
                        ha='center', va='bottom', color='green', fontsize=10)
                pdf_lower = norm.pdf(lower_ci, mean_val, se)
                pdf_upper = norm.pdf(upper_ci, mean_val, se)
                ax.text(lower_ci, pdf_lower * 1.05, f"Lower CI: {lower_ci:.2f}",
                        ha='center', va='bottom', color='red', fontsize=10)
                ax.text(upper_ci, pdf_upper * 1.05, f"Upper CI: {upper_ci:.2f}",
                        ha='center', va='bottom', color='red', fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(title="Variant", fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.show()

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
