# step 1: analysis of representative data

# data cleaning - removal of all NaN scores
# produces CSV file with all clean scores + CSV with all NaN and failed values 
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml/data
mkdir combined_scores
cd /Users/shreyasree/Documents/GitHub/KEAP1_ml
python3 scripts/representative_analysis/data_cleaning.py --input data/docking_scores_representatives --output data/combined_scores/combined_valid_scores.csv --failed data/combined_scores/combined_failed_scores.csv

# data cleaning - representative data 
# generates a file of combined cluster representative details and its docking score 
python3 scripts/representative_analysis/representative_analysis.py --reps data/cluster_representatives_7_per_cluster.csv --scores data/combined_scores/combined_valid_scores.csv --output data/representative_docking_scores

# generates a histogram of the distribution of docking scores
Rscript scripts/representative_analysis/visualization/docking_score_dist.R data/combined_scores/combined_valid_scores.csv Score plots/representative_analysis/docking_score_histogram.pdf 0.5

# generating a boxplot of distribution of representative analysis scores 
Rscript scripts/representative_analysis/visualization/representative_analysis_dist.R data/representative_docking_scores plots/representative_analysis/representative_score_dist_analysis.pdf "Docking Affinity Distribution of Representative Molecules against KEAP1"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------