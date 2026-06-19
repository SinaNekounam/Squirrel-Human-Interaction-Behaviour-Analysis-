import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# get preprocessed data from csv file

data_path = 'task_1_preprocessed_squirrel_data.csv'
merged = pd.read_csv(data_path)

# select four feature scores from preprocessing and set them as behaviour features
behaviour_features = ['active_score', 'foraging_score', 'vocal_score', 'alert_score']

# select environment features
environment_features = ['X', 'Y', 'conditions_enc', 'has_threat', 'total_squirrels']


print("\n---------------------------------------------------------------------------")
print("MISSING VALUES CHECK")

# check missing values in behaviour features and environment features
print('Missing values in behaviour features:')
print(merged[behaviour_features].isnull().sum())

print('\nMissing values in environment features:')
print(merged[environment_features].isnull().sum())

# drop missing values if any exist
merged = merged.dropna(subset=behaviour_features + environment_features)
print(f'\nDataset size after dropping missing values: {len(merged)}')



# CLUSTERING 1：Behaviour Clustering
print("\n---------------------------------------------------------------------------")
print("BEHAVIOUR CLUSTERING")

X_behaviour = merged[behaviour_features]

# normalise data by minmanscaler, keep same range
scaler = MinMaxScaler()
X_behaviour_scaled = scaler.fit_transform(X_behaviour)

# create elbow curve to determine optimal k
inertias = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_behaviour_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, "o-")
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors')
plt.title('Elbow Method - Behaviour Clustering')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('task4_elbow_1.png')
plt.close()

print('\ntask4_elbow_1.png was saved')


# KMeans clustering
optimal_k_1 = 5
km_behaviour = KMeans(n_clusters=optimal_k_1, random_state=42)
merged['behaviour_cluster'] = km_behaviour.fit_predict(X_behaviour_scaled)

# check values
print('\nBehaviour Cluster Distribution:')
print(merged['behaviour_cluster'].value_counts().sort_index())

# heatmap 1, show each cluster using mean feature values
cluster_means = merged.groupby('behaviour_cluster')[behaviour_features].mean()

plt.figure(figsize=(8, 5))
sns.heatmap(cluster_means, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Behaviour Score Means per Cluster')
plt.xlabel('Feature')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('task4_heatmap_feat_mean_1.png')
plt.close()

print('\ntask4_heatmap_feat_mean_1.png was saved')

# based on heatmpa, describe each cluster
# Cluster 0: Forager          - high foraging(1.00)
# Cluster 1: Active           - high active(1.00)
# Cluster 2: Alert Forager    - high foraging(1.01) and high alert(1.05)
# Cluster 3: Most Forager     - very high foraging(2.00)
# Cluster 4: Alert Active     - high active(1.02) and high alert(1.03)

# heatmap 2, cross with human_reaction to answer the research question
ct = pd.crosstab(merged['behaviour_cluster'],
                 merged['human_reaction'],
                 normalize='index')

plt.figure(figsize=(8, 5))
sns.heatmap(ct, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Runs from', 'Indifferent', 'Approaches'],
            yticklabels=['Forager', 'Active', 'Alert Forager',
                        'Most Forager', 'Alert Active'])
plt.title('Human Reaction Distribution per Behaviour Cluster')
plt.xlabel('Human Reaction')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('task4_human_reaction_1.png')
plt.close()

print('\ntask4_human_reaction_1.png was saved')


# CLUSTERING 2：Environment Clustering
print("\n---------------------------------------------------------------------------")
print("ENVIRONMENT CLUSTERING")

X_environment = merged[environment_features]

# normalise by StandardScaler
scaler2 = StandardScaler()
X_env_scaled = scaler2.fit_transform(X_environment)

# elbow method to find optimal k
inertias2 = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_env_scaled)
    inertias2.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias2, "o-")
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors')
plt.title('Elbow Method - Environment Clustering')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('task4_elbow_2.png')
plt.close()

print('\ntask4_elbow_2.png was saved')

# KMeans clustering
optimal_k_2 = 3
km_env = KMeans(n_clusters=optimal_k_2, random_state=42)
merged['env_cluster'] = km_env.fit_predict(X_env_scaled)

print('\nEnvironment Cluster Distribution:')
print(merged['env_cluster'].value_counts().sort_index())

# scatter plot show clusters' observation distribution
plt.figure(figsize=(10, 6))
plt.scatter(merged['X'], merged['Y'],
            c=merged['env_cluster'],
            alpha=0.3, s=10)
plt.colorbar(label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Environment Clusters - Geographic Distribution')
plt.tight_layout()
plt.savefig('task4_scatter_env_clusters.png')
plt.close()

print('\ntask4_scatter_env_clusters.png was saved')

# heatmap 3, show each cluster using mean feature values
env_feature_cols = ['conditions_enc', 'has_threat', 'total_squirrels']
cluster_means2 = merged.groupby('env_cluster')[env_feature_cols].mean()

plt.figure(figsize=(8, 5))
sns.heatmap(cluster_means2, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Environment Feature Means per Cluster')
plt.xlabel('Feature')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('task4_heatmap_feat_mean_2.png')
plt.close()

print('\ntask4_heatmap_feat_mean_2.png was saved')

# based on heatmap 3, define each cluster
# Cluster 0: High-Threat Low-Density     - high threat(0.88)，low density(5.56), relatively busy condition
# Cluster 1: Medium Environment          - medium threat(0.69)，relatively medium density(6.32), relatively medium condition
# Cluster 2: High-Density Busy           - relatively low threat(0.56)，very high density(12.53)，busy condition

# heatmap 4, cross with human_reaction to answer research question
ct2 = pd.crosstab(merged['env_cluster'],
                  merged['human_reaction'],
                  normalize='index')

plt.figure(figsize=(8, 5))
sns.heatmap(ct2, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Runs from', 'Indifferent', 'Approaches'],
            yticklabels=['High-Threat Low-Density', 'Medium Environment', 'High-Density Busy'])
plt.title('Human Reaction Distribution per Environment Cluster')
plt.xlabel('Human Reaction')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('task4_human_reaction_2.png')
plt.close()

print('\ntask4_human_reaction_2.png was saved')


print("\n---------------------------------------------------------------------------")
print("All clustering outputs were saved successfully.")
