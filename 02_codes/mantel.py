import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import mantel
import random
import os

# directories
data_dir = '../01_data/'
output_dir = '../03_results/out_mantel'
os.makedirs(output_dir, exist_ok=True)

phylum_filename = 'phylum_count.tsv'
genus_filename = 'genus_count.tsv'
order_filename = 'order_count.tsv'

phylum_df_path = os.path.join(data_dir, phylum_filename)
genus_df_path = os.path.join(data_dir, genus_filename)
order_df_path = os.path.join(data_dir, order_filename)

# read dataframes
phylum_df = pd.read_csv(phylum_df_path, sep='\t', index_col=0)
genus_df = pd.read_csv(genus_df_path, sep='\t', index_col=0)
order_df = pd.read_csv(order_df_path, sep='\t', index_col=0)

def perform_mantel_test(df, group1_prefix, group2_prefix):
    # Extract rows based on the prefixes
    group1_rows = [row for row in df.index if row.startswith(group1_prefix)]
    group2_rows = [row for row in df.index if row.startswith(group2_prefix)]
    
    if len(group1_rows) > len(group2_rows): # check if group1 has more samples than group2, and if so, randomly remove samples from group1
        num_to_remove = len(group1_rows) - len(group2_rows)
        group1_rows = random.sample(group1_rows, len(group2_rows))  # Select random samples from group1
    
    group1_matrix = df.loc[group1_rows]
    group2_matrix = df.loc[group2_rows]
    
    group1_distances = pdist(group1_matrix.values, metric='euclidean')
    group2_distances = pdist(group2_matrix.values, metric='euclidean')
    
    group1_distance_matrix = squareform(group1_distances)
    group2_distance_matrix = squareform(group2_distances)
    
    # mantel test
    mantel_stat, mantel_p_value, _ = mantel(group1_distance_matrix, group2_distance_matrix)
    
    return mantel_stat, mantel_p_value

# function to perform mantel test
def run_mantel_tests_and_save_results(group1_prefix, group2_prefix, results_list):
    datasets = {'Phylum': phylum_df, 'Genus': genus_df, 'Order': order_df}
    for dataset_name, df in datasets.items():
        stat, p_value = perform_mantel_test(df, group1_prefix, group2_prefix)
        results_list.append([dataset_name, group1_prefix, group2_prefix, stat, p_value])

mantel_results = []

# Run tests and collect results
run_mantel_tests_and_save_results('AC', 'AS', mantel_results)
run_mantel_tests_and_save_results('EC', 'ES', mantel_results)
run_mantel_tests_and_save_results('AC', 'EC', mantel_results)
run_mantel_tests_and_save_results('AS', 'ES', mantel_results)

results_df = pd.DataFrame(mantel_results, columns=['Dataset', 'Group 1 Prefix', 'Group 2 Prefix', 'Mantel Statistic', 'P-value'])

results_filename = 'mantel_results.tsv'
out_filename = os.path.join(output_dir, results_filename)
results_df.to_csv(out_filename, sep='\t', index=False)

print("Mantel test results saved to Mantel_results.tsv")