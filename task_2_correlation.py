# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. LOAD DATA

# read the preprocessed data
clean_data = pd.read_csv("task_1_preprocessed_squirrel_data.csv")


# 2. INSPECT DATA
# get the size
data_size = clean_data.shape

# see the first few rows
clean_data.head(10)

# preview rows
clean_data.columns

# check datatypes
# clean_data.info()


# 3. TARGET DISTRIBUTION

# 3.1 Count target categories
# value_counts() counts how many times each interaction category appears
# 3.1 Count target categories
# value_counts() counts how many times each interaction category appears
human_interaction_counts = clean_data["human_reaction"].map({
    0: "avoid",
    1: "neutral",
    2: "approach"
}).value_counts()

human_interaction_counts = human_interaction_counts[["neutral", "avoid", "approach"]]

print("\n---------------------------------------------------------------------------")
print("Target distribution:")
print(human_interaction_counts)


# 3.2 Calculate target percentages
# normalize = True turns counts into proportions, then * 100 converts them to percentages
human_interaction_counts_percentage = clean_data["human_reaction"].map({
    0: "avoid",
    1: "neutral",
    2: "approach"
}).value_counts(normalize=True) * 100

human_interaction_counts_percentage = human_interaction_counts_percentage[["neutral", "avoid", "approach"]]

# Round percentages to 2 decimal places
human_interaction_counts_percentage_rounded = human_interaction_counts_percentage.round(2)

print("\n---------------------------------------------------------------------------")
print("Target distribution percentage:")
print(human_interaction_counts_percentage_rounded)


# 3.3 Visualise target distribution
plt.figure(figsize=(8, 5))

# Plot the already-counted interaction categories
human_interaction_counts.plot(kind="bar")

plt.xticks(rotation=0)
plt.xlabel("Interaction Type")
plt.ylabel("Number of Observations")
plt.title("Distribution of Human Interaction Categories")
plt.tight_layout()

# Save the figure
print("\n---------------------------------------------------------------------------")
plt.savefig("/home/task_2_distribution_of_human_interaction_categories.jpg")
print("task_2_distribution_of_human_interaction_categories bar chart was saved")


# 4. ENCODE HUMAN INTERACTION AND CATEGORICAL FEATURES

# 4.1 create map for human_interaction
# human_reaction is already numeric:
# avoid = 0
# neutral = 1
# approach = 2

# 4.2 create interaction_numeric column
clean_data["human_interaction_numeric"] = clean_data["human_reaction"]

# 4.3 create map for Shift
shift_map = {
    "AM": 0,
    "PM": 1
}

# 4.4 create shift_numeric column
clean_data["shift_numeric"] = clean_data["Shift"].map(shift_map)

# 4.5 check new columns
# check if the new clumns are right
clean_data[[ "Shift", "shift_numeric" ]].head()

clean_data[[ "human_reaction", "human_interaction_numeric" ]].head()


# 5. SELECT CORRELATION FEATURES

# 5.1 define useful features
behaviour_features = ["Running", "Chasing", "Climbing", "Eating", "Foraging", "Kuks", "Quaas", "Moans",
                      "Tail flags", "Tail twitches", "active_score", "foraging_score", "vocal_score", "alert_score"]

location_features = ["X", "Y", "total_squirrels", "conditions_enc", "has_threat"]

time_features = ["shift_numeric"]

target_feature = ["human_interaction_numeric"]

# 5.2 exclude useless columns

# These columns are excluded from correlation analysis because they are identifiers,
# raw text fields, or categorical variables that are not suitable for simple numeric correlation.

excluded_features = ["Unique Squirrel ID", "Hectare", "Age", "Primary Fur Color", "Highlight Fur Color", "Combination of Primary and Highlight Color",
                     "Color notes", "Location", "Above Ground Sighter Measurement", "Specific Location", "Other Activities", "Other Interactions","Lat/Long"]

# 5.3 store selected columns in list
# combine all features lists
selected_features = target_feature + behaviour_features + location_features + time_features

# 5.4 build smaller correlation dataframe
corr_data = clean_data[selected_features]
print("\n---------------------------------------------------------------------------")
print("Correlation Datafram was made.")
#print(corr_data.head(10))

# 5.5 inspect resulting dataframe
print("\n---------------------------------------------------------------------------")
print("Correlation dataframe size:", corr_data.shape)

print("\n---------------------------------------------------------------------------")
print("First 5 rows of correlation dataframe:")
print(corr_data.head(5))

print("\n---------------------------------------------------------------------------")
print("Correlation dataframe information:")
corr_data.info()


# 6. HANDLE MISSING VALUES

# 6.1 check missing values
# count how many missing values each column contains
# isna() is a function that asks is this value missing?
missing_values_count = corr_data.isna().sum()

# 6.2 measure severity
# look at the output and ask are missing values small or large?

# 6.3 choose strategy
# our strategy: remove rows containing missing values

# 6.4 clean correlation dataframe

# remove rows containing NaN values
corr_data_clean = corr_data.dropna()
print("\n---------------------------------------------------------------------------")
print("Remove rows containing values that are not a number")
print(corr_data_clean.head(10))

# 6.5 verify results

# Check dataframe size after removing missing values
print("\n---------------------------------------------------------------------------")
print("Check dataframe size after removing missing values")
print("Original correlation dataframe size:", corr_data.shape)
print("Cleaned correlation dataframe size:", corr_data_clean.shape)

# Check if any missing values still remain
print("\n---------------------------------------------------------------------------")
print("Remaining missing values after cleaning:")
print(corr_data_clean.isna().sum())


# 7. COMPUTE CORRELATION MATRIX
# How strongly are features related to each other?

# 7.1 compute Pearson correlations
# How strongly is every feature related to every other feature?

# 7.2 store pearson correlation
# this code is in week 7 tutorial
pearson_correlations = corr_data_clean.corr(method = "pearson")

# 7.3 inspect relationships
# print the result
print("\n---------------------------------------------------------------------------")
print("Pearson Correlation: ")
print(pearson_correlations)

# 7.4 focus on target relationships
# now we want to isolate all correlations with human_interaction_numeric
# Extract correlations with the target variable
interaction_correlations = pearson_correlations["human_interaction_numeric"]
print("\n---------------------------------------------------------------------------")
print("Extract correlations with the target variable")
print(interaction_correlations)


# 8. EXTRACT TARGET CORRELATIONS

# 8.1 extract target relationships
# Select only the correlations between each feature and the target variable
target_correlations = pearson_correlations["human_interaction_numeric"]

print("\n---------------------------------------------------------------------------")
print("Correlations with human_interaction_numeric:")
print(target_correlations)

# 8.2 remove meaningless self-correlation
# You want to remove human_interaction_numeric because it is meaningless self-correlation
target_correlations = target_correlations.drop("human_interaction_numeric")

print("\n---------------------------------------------------------------------------")
print("Correlations without human_interaction_numeric:")
print(target_correlations)

# 8.3 sort strongest relationships
target_correlations = target_correlations[
    [
        "foraging_score",
        "Foraging",
        "Eating",
        "total_squirrels",
        "conditions_enc",
        "Chasing",
        "Tail twitches",
        "has_threat",
        "Tail flags",
        "alert_score",
        "Quaas",
        "Moans",
        "shift_numeric",
        "Kuks",
        "vocal_score",
        "Climbing",
        "X",
        "active_score",
        "Running",
        "Y"
    ]
]

print("\n---------------------------------------------------------------------------")
print("Correlations sorted:")
print(target_correlations)

# 8.4 interpret influential features
# Features such as foraging_score and Foraging show weak positive
# correlations with human interaction behaviour.
# Running and active_score show weak negative correlations.
# Most correlations are relatively weak, suggesting squirrel-human
# interaction behaviour is influenced by multiple factors.


# 9. VISUALISE RESULTS

# 9.1 target correlations
# Use the sorted target correlations from Step 8 for visualization

# 9.2 create readable bar chart
plt.figure(figsize=(12, 6))

# Plot the target correlations as a bar chart
target_correlations.plot(kind="bar")

# 9.3 Highlight positive/negative relationships
# Add a horizontal line at 0 to separate positive and negative correlations
plt.axhline(0)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Pearson Correlation Coefficient")
plt.title("Feature Correlations with Human Interaction Behaviour")
plt.tight_layout()

# 9.4 save visualization
print("\n---------------------------------------------------------------------------")
plt.savefig("/home/task_2_feature_correlations_with_human_interaction_behaviour.jpg")
print("task_2_feature_correlations_with_human_interaction_behaviour bar chart was saved")

# 9.5 interpret behavioural patterns
# the strongest features are: forging_score, foraging, eating
# the weakest are: Y, X, Running, active score

# Howver, overall, the correlations were relatively weak, suggesting that squirrel-human interaction behaviour is likely
# influenced by multiple behavioural and environmental factors rather than a single dominant feature.


# 10. SAVE OUTPUTS

# Save sorted target correlations as a CSV file
target_correlations.to_csv(
    "/home/task_2_target_correlations.csv",
    header=["Pearson Correlation"]
)

print("\n---------------------------------------------------------------------------")
print("Sorted target correlations was saved as a CSV file")
