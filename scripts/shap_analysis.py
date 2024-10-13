import shap


def perform_shap_analysis(model, X_train, X_test, feature_names):
    """Perform SHAP analysis and plot the SHAP values."""
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    shap_values_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_test,
                                       feature_names=feature_names)

    shap.plots.beeswarm(shap_values_exp, max_display=50)
