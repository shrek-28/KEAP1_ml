library(tidyverse)

plot_top20_shap <- function(file_path,
                            model_name,
                            plot_title = NULL,
                            output_path = NULL,
                            width = 10,
                            height = 8) {
  
  df <- read.csv(file_path)
  
  df$MeanAbsSHAP <- as.numeric(df$MeanAbsSHAP)
  
  top20 <- df %>%
    arrange(desc(MeanAbsSHAP)) %>%
    slice_head(n = 20)
  
  if (is.null(plot_title)) {
    plot_title <- paste("Top 20 SHAP Features -", model_name)
  }
  
  p <- ggplot(top20, aes(
    x = reorder(Feature, MeanAbsSHAP),
    y = MeanAbsSHAP
  )) +
    geom_bar(stat = "identity", fill = "#FF46A2") +
    
    # value labels on the right of bars
    geom_text(
      aes(label = round(MeanAbsSHAP, 3)),
      hjust = -0.1,
      size = 3
    ) +
    
    coord_flip() +
    
    # expand limits so labels don't get clipped
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    
    labs(
      title = plot_title,
      x = "Feature",
      y = "Mean |SHAP|"
    ) +
    theme_minimal()
  
  print(p)
  
  if (!is.null(output_path)) {
    ggsave(output_path, plot = p, device = "pdf",
           width = width, height = height)
  }
  
  return(top20)
}

plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/lin_reg/shap_importance.csv", "Linear Regression", plot_title = "SHAP Feature Importances: Linear Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/linear_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/lasso_reg/lasso_shap_feature_importance.csv", "Lasso Regression", plot_title = "SHAP Feature Importances: Lasso Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/lasso_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/ridge_regressor/ridge_shap_feature_importance.csv", "Ridge Regression", plot_title = "SHAP Feature Importances: Ridge Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/ridge_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/elastic_net_regressor/elasticnet_shap_feature_importance.csv", "Elastic Net Regression", plot_title = "SHAP Feature Importances: Elastic Net Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/elastic_net_reg.pdf")

plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/knn_regressor/knn_shap_feature_importance.csv", "KNN Regression", plot_title = "SHAP Feature Importances: K-Nearest Neighbours Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/knn_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/svm_regressor/svr_shap_feature_importance.csv", "SVM Regression", plot_title = "SHAP Feature Importances: Support Vector Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/svm_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/poly_regressor/poly_shap_feature_importance.csv", "Polynomial Regression", plot_title = "SHAP Feature Importances: Polynomial Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/polynomial_reg.pdf")

plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/decision_tree_regressor/decision_tree_shap_feature_importance.csv", "DT Regression", plot_title = "SHAP Feature Importances: Decision Tree Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/decision_tree_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/random_forest_regressor/rf_shap_feature_importance.csv", "rf Regression", plot_title = "SHAP Feature Importances: Random Forest Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/random_forest_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/gradient_boosting_regressor/gbr_shap_feature_importance.csv", "GBR Regression", plot_title = "SHAP Feature Importances: Gradient Boosting Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/grad_boost_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/adaboost_regressor/shap_importance.csv", "AdaBoost Regression", plot_title = "SHAP Feature Importances: AdaBoost Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/adaboost_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/catboost_regressor/catboost_shap_feature_importance.csv", "CatBoost Regression", plot_title = "SHAP Feature Importances: CatBoost Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/catboost_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/xgboost_regressor/xgboost_shap_feature_importance.csv", "XGBoost Regression", plot_title = "SHAP Feature Importances: XGBoost Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/xgboost_reg.pdf")
plot_top20_shap("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/SHAP/lightgbm_regressor/lightgbm_shap_feature_importance.csv", "LightGBM Regression", plot_title = "SHAP Feature Importances: LightGBM Regression", output_path="/Users/shreyasree/Documents/GitHub/KEAP1_ml/plots/SHAP/lightgbm_reg.pdf")

