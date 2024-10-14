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
nutrients_filename = 'nutrients_all.tsv'

phylum_df_path = os.path.join(data_dir, phylum_filename)
genus_df_path = os.path.join(data_dir, genus_filename)
order_df_path = os.path.join(data_dir, order_filename)
nutrients_df_path = os.path.join(data_dir, nutrients_filename)

# read dataframes
phylum_df = pd.read_csv(phylum_df_path, sep='\t', index_col=0)
genus_df = pd.read_csv(genus_df_path, sep='\t', index_col=0)
order_df = pd.read_csv(order_df_path, sep='\t', index_col=0)
nutrients_df = pd.read_csv(nutrients_df_path, sep='\t', index_col=0)

def perform_mantel_test_between_datasets(df1, df2, prefix):
    # Select samples (rows) that start with the prefix
    df1_samples = [row for row in df1.index if row.startswith(prefix)]
    df2_samples = [row for row in df2.index if row.startswith(prefix)]
    
    # Find the common samples
    common_samples = set(df1_samples).intersection(df2_samples)
    
    if not common_samples:
        print(f"No hay muestras comunes para el prefijo {prefix}")
        return None, None
    
    # Subset the dataframes to the common samples
    df1_subset = df1.loc[common_samples]
    df2_subset = df2.loc[common_samples]
    
    # Compute distance matrices
    df1_distances = pdist(df1_subset.values, metric='euclidean')
    df2_distances = pdist(df2_subset.values, metric='euclidean')
    
    df1_distance_matrix = squareform(df1_distances)
    df2_distance_matrix = squareform(df2_distances)
    
    # Perform Mantel test
    mantel_stat, mantel_p_value, _ = mantel(df1_distance_matrix, df2_distance_matrix)
    
    return mantel_stat, mantel_p_value

def run_mantel_tests_between_motus_and_nutrients(prefixes, results_list):
    datasets = {
        'Phylum': phylum_df, 
        'Genus': genus_df, 
        'Order': order_df,
    }
    
    for prefix in prefixes:
        for dataset_name, df in datasets.items():
            stat, p_value = perform_mantel_test_between_datasets(df, nutrients_df, prefix)
            if stat is not None:
                results_list.append([dataset_name, prefix, stat, p_value])
            else:
                print(f"No hay muestras comunes para el conjunto de datos {dataset_name} y el prefijo {prefix}")

mantel_results = []

# Run tests and collect results
run_mantel_tests_between_motus_and_nutrients(['AS', 'ES'], mantel_results)

# Convert results to a DataFrame
results_df = pd.DataFrame(mantel_results, columns=['Dataset', 'Prefix', 'Mantel Statistic', 'P-value'])

# Save results to a TSV file
results_filename = 'mantel_results_motus_vs_nutrients.tsv'
out_filename = os.path.join(output_dir, results_filename)
results_df.to_csv(out_filename, sep='\t', index=False)

print(f"Resultados de las pruebas de Mantel guardados en {out_filename}")
