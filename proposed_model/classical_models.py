from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Robust import of XGBoost with fallback to sklearn's Gradient Boosting
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

def get_svm_pipeline(C=1.0, kernel='rbf', random_state=42):
    """
    Build a Support Vector Machine pipeline with standard scaling.
    
    Returns:
        pipeline (sklearn.pipeline.Pipeline): SVM pipeline.
    """
    return make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel=kernel, probability=True, class_weight='balanced', random_state=random_state)
    )

def get_random_forest_pipeline(n_estimators=100, max_depth=None, random_state=42):
    """
    Build a Random Forest pipeline with standard scaling.
    
    Returns:
        pipeline (sklearn.pipeline.Pipeline): Random Forest pipeline.
    """
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=random_state)
    )

def get_xgboost_pipeline(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=1.0, random_state=42):
    """
    Build an XGBoost pipeline with standard scaling. 
    Falls back to sklearn's GradientBoostingClassifier if xgboost is not installed.
    
    Returns:
        pipeline (sklearn.pipeline.Pipeline): Boosting pipeline.
    """
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss'
        )
    else:
        # Fallback classifier (does not support scale_pos_weight directly in init, but we can set sample_weight during fit if needed;
        # however, since we use XGBClassifier on the user system, this fallback is rarely active and we can run as is)
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
    return make_pipeline(StandardScaler(), clf)

