import warnings
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import streamlit as st
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC
from xgboost import XGBClassifier
import optuna
import shap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', None)

# --- Veri Ön İşleme Fonksiyonları ---
def ml_features_drop(df, corr_threshold=0.8, threshold=0.1):
    df_clean = df.copy()
    silinecek_sutunlar = ['comments', 'Country', 'state', 'Timestamp', 'UserID', 'SurveyID']
    df_clean = df_clean.drop(columns=silinecek_sutunlar, errors='ignore')

    numeric_df = df_clean.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]

    if to_drop_corr:
        df_clean = df_clean.drop(columns=to_drop_corr)

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)
    low_var_cols = numeric_df.columns[~selector.get_support()]

    if len(low_var_cols) > 0:
        df_clean = df_clean.drop(columns=low_var_cols)

    return df_clean

def ml_fill_missing_values(df_clean):
    temiz_yaslar = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 75)]
    yas_medyani = temiz_yaslar['Age'].median()
    df_clean['Age'] = df_clean['Age'].fillna(yas_medyani)
    df_clean.loc[(df_clean['Age'] < 18) | (df_clean['Age'] > 75), 'Age'] = yas_medyani

    sirket_kolonlari = [
        'self_employed', 'family_history', 'mental_health_consequence',
        'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
        'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'work_interfere',
        'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
        'wellness_program', 'seek_help', 'anonymity', 'leave'
    ]

    for kolon in sirket_kolonlari:
        en_cok_tekrar_eden = df_clean[kolon].mode()[0]
        df_clean[kolon] = df_clean[kolon].fillna(en_cok_tekrar_eden)

    df_clean['Gender'] = df_clean['Gender'].fillna('Other/Prefer Not To Say')
    return df_clean

def nominal_to_ordinal(df):
    df_encoded = df.copy()
    gender_map = {
        'Male': 0, 'Female': 1, 'Non-Binary / Trans': 2, 'Non-binary': 2,
        'Other / Prefer not to say': 3, 'Other/Prefer Not To Say': 3,
        'Other / Prefer Not To Say': 3, 'Other': 3,
    }
    df_encoded['Gender'] = df_encoded['Gender'].map(gender_map)

    employee_map = {'1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000': 5, 'More than 1000': 6}
    df_encoded['no_employees'] = df_encoded['no_employees'].map(employee_map)

    self_employed_map = {'No': 0, '0.0': 0, 0.0: 0, 'Yes': 1, '1.0': 1, 1.0: 1}
    df_encoded['self_employed'] = df_encoded['self_employed'].map(self_employed_map)

    family_history_map = {'No': 0, "I don't know": 0.5, "Don't know": 0.5, 'Yes': 1}
    df_encoded['family_history'] = df_encoded['family_history'].map(family_history_map)

    obs_consequence_map = {'No': 0, 'Yes': 1}
    df_encoded['obs_consequence'] = df_encoded['obs_consequence'].map(obs_consequence_map)

    consequence_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
    df_encoded['mental_health_consequence'] = df_encoded['mental_health_consequence'].map(consequence_map)
    df_encoded['phys_health_consequence'] = df_encoded['phys_health_consequence'].map(consequence_map)

    interview_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
    df_encoded['mental_health_interview'] = df_encoded['mental_health_interview'].map(interview_map)
    df_encoded['phys_health_interview'] = df_encoded['phys_health_interview'].map(interview_map)

    support_people_map = {'No': 0, 'Maybe': 1, 'Some of them': 1, 'Yes': 2}
    df_encoded['coworkers'] = df_encoded['coworkers'].map(support_people_map)
    df_encoded['supervisor'] = df_encoded['supervisor'].map(support_people_map)

    work_interfere_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    df_encoded['work_interfere'] = df_encoded['work_interfere'].map(work_interfere_map)

    remote_work_map = {'No': 0, 'Never': 0, 'Sometimes': 1, 'Yes': 2, 'Always': 2}
    df_encoded['remote_work'] = df_encoded['remote_work'].map(remote_work_map)

    tech_company_map = {'No': 0, '0.0': 0, 0.0: 0, 'Yes': 1, '1.0': 1, 1.0: 1, "Don't know": 0}
    df_encoded['tech_company'] = df_encoded['tech_company'].map(tech_company_map)

    benefit_awareness_map = {'No': 0, 'Not eligible for coverage / NA': 0, "Don't know": 0.5, "I don't know": 0.5, 'Yes': 1}
    df_encoded['benefits'] = df_encoded['benefits'].map(benefit_awareness_map)
    df_encoded['wellness_program'] = df_encoded['wellness_program'].map(benefit_awareness_map)
    df_encoded['seek_help'] = df_encoded['seek_help'].map(benefit_awareness_map)
    df_encoded['anonymity'] = df_encoded['anonymity'].map(benefit_awareness_map)
    df_encoded['mental_vs_physical'] = df_encoded['mental_vs_physical'].map(benefit_awareness_map)

    care_options_map = {'No': 0, 'Not sure': 0.5, 'I am not sure': 0.5, 'Yes': 1}
    df_encoded['care_options'] = df_encoded['care_options'].map(care_options_map)

    leave_map = {
        'Very easy': 0, 'Somewhat easy': 1, "Don't know": 2, "I don't know": 2,
        'Neither easy nor difficult': 2, 'Somewhat difficult': 3, 'Difficult': 4, 'Very difficult': 4
    }
    df_encoded['leave'] = df_encoded['leave'].map(leave_map)
    return df_encoded

def unsupervised_scaling(df_encoded):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    return pd.DataFrame(X_scaled, columns=df_encoded.columns, index=df_encoded.index)

def apply_pca(df_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    pca_data = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(n_components)])
    total_variance = sum(pca.explained_variance_ratio_) * 100
    return df_pca, total_variance

def calculate_kmeans(df_scaled):
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d_array = pca_2d.fit_transform(df_scaled)
    var_2d = np.sum(pca_2d.explained_variance_ratio_) * 100
    df_pca_2d = pd.DataFrame(pca_2d_array, columns=['PC1', 'PC2'])

    setattr(KMeans, "_estimator_type", "clusterer")

    model_elbow = KMeans(random_state=42, n_init='auto')
    elbow = KElbowVisualizer(model_elbow, k=(2, 10), metric='distortion')
    elbow.fit(df_pca_2d)
    elbow.finalize()
    fig_elbow = elbow.fig
    optimal_k = elbow.elbow_value_ if elbow.elbow_value_ is not None else 4

    plt.figure()

    model_silhouette = KMeans(random_state=42, n_init='auto')
    silhouette_vis = KElbowVisualizer(model_silhouette, k=(2, 10), metric='silhouette')
    silhouette_vis.fit(df_pca_2d)
    silhouette_vis.finalize()
    fig_silhouette = silhouette_vis.fig

    plt.close('all')
    return fig_elbow, fig_silhouette, optimal_k, df_pca_2d, var_2d

def get_cluster_profiles(df_original, df_pca, kmeans_model):
    df_result = df_original.copy()
    df_result['Cluster'] = kmeans_model.labels_
    numeric_cols = df_result.select_dtypes(include=['number']).columns
    cluster_summary = df_result.groupby('Cluster')[numeric_cols].mean().round(2)
    cluster_summary['Kullanici_Sayisi'] = df_result['Cluster'].value_counts().sort_index()
    return df_result, cluster_summary

def plot_dendrogram(df_scaled, n_clusters=4):
    linked = linkage(df_scaled, method='ward')
    distances = linked[:, 2]
    threshold = (distances[-n_clusters + 1] + distances[-n_clusters]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    dendrogram(linked, orientation='top', color_threshold=threshold, distance_sort='descending', show_leaf_counts=False, ax=ax)
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'K={n_clusters} Kesim (Mesafe ≈ {round(threshold, 1)})')
    ax.set_title(f'Hiyerarşik Kümeleme - Dendrogram (K={n_clusters})', fontsize=10)
    ax.set_xlabel('Veri Noktaları')
    ax.set_ylabel('Mesafe (Distance)')
    ax.legend(loc='upper right', fontsize=8)
    return fig

def calculate_hierarchical(df_scaled, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(df_scaled)
    return labels, model

# --- Supervised Learning Bölümü ---
def read_train():
    try:
        osmi_train = pd.read_csv("dataset/mental_health_train.csv")
        X_train_ml = osmi_train.drop(columns=["treatment", "turnover_risk"])
        y_train_treatment = osmi_train["treatment"]
        y_train_turnover = osmi_train["turnover_risk"]
        return X_train_ml, y_train_treatment, y_train_turnover
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None, None

def ml_cross_validation(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM (SVC)': CalibratedClassifierCV(SVC(random_state=42), ensemble=False),        
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    }

    scoring = {
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc_ovr',
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_results = []

    for name, model in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        baseline_results.append({
            'Model': name,
            'Precision (Mean)': cv_results['test_precision'].mean(),
            'Recall (Mean)': cv_results['test_recall'].mean(),
            'F1-Score (Mean)': cv_results['test_f1'].mean(),
            'ROC-AUC (Mean)': cv_results['test_roc_auc'].mean(),
        })

    df_results = pd.DataFrame(baseline_results).sort_values(by='F1-Score (Mean)', ascending=False).reset_index(drop=True)
    return df_results

# --- Optuna & Best Parametre Analiz Metotları ---
def optimize_all_models_with_optuna(X_train, y_train, n_trials=15):
    """
    Seçili modeller için Optuna optimizasyonu çalıştırarak her modele özel en iyi parametreleri bulur.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_to_optimize = ["Random Forest", "LightGBM", "XGBoost", "CatBoost", "Gradient Boosting", "Logistic Regression"]    
    tuned_results = {}

    for model_name in models_to_optimize:
        def objective(trial):
            if model_name == "Random Forest":
                params = {
                    #'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    #'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=25),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                    'random_state': 42, #'n_jobs': -1
                }
                clf = RandomForestClassifier(**params)

            elif model_name == "LightGBM":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=25),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'random_state': 42, 'verbose': -1, #'n_jobs': -1
                }
                clf = LGBMClassifier(**params)

            elif model_name == "XGBoost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=25),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'random_state': 42, 'eval_metric': 'logloss', #'n_jobs': -1
                }
                clf = XGBClassifier(**params)

            elif model_name == "Logistic Regression":
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'max_iter': 1000, 'random_state': 42
                }
                clf = LogisticRegression(**params)
            elif model_name == "CatBoost":
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 150),
                    'depth': trial.suggest_int('depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                    'random_state': 42, 'verbose': 0
                }
                clf = CatBoostClassifier(**params)

            elif model_name == "Gradient Boosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=25),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                    'random_state': 42
                }
                clf = GradientBoostingClassifier(**params)

            scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
            return scores.mean()

        #study = optuna.create_study(direction="maximize")
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
        )
        study.optimize(objective, n_trials=n_trials)

        tuned_results[model_name] = {
            'best_params': study.best_params,
            'best_cv_f1': study.best_value
        }

    return tuned_results

def evaluate_all_tuned_models(X_train, y_train, tuned_params_dict):
    """
    Optuna parametreleri uygulanmış modelleri Cross-Validation'a sokar,
    Train vs CV (Validation) farkı üzerinden Overfit riskini ölçer.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {'f1': 'f1_macro', 'roc_auc': 'roc_auc_ovr'}
    results = []

    for model_name, data in tuned_params_dict.items():
        params = data['best_params']

        if model_name == "Random Forest":
            clf = RandomForestClassifier(**params)
        elif model_name == "LightGBM":
            clf = LGBMClassifier(**params, verbose=-1)
        elif model_name == "XGBoost":
            clf = XGBClassifier(**params, eval_metric='logloss')
        elif model_name == "Logistic Regression":
            clf = LogisticRegression(**params)
        elif model_name == "CatBoost":
            clf = CatBoostClassifier(**params, verbose=0)
        elif model_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(**params)

        cv_res = cross_validate(clf, X_train, y_train, cv=skf, scoring=scoring, return_train_score=True, n_jobs=-1)

        train_f1 = cv_res['train_f1'].mean()
        val_f1 = cv_res['test_f1'].mean()
        train_auc = cv_res['train_roc_auc'].mean()
        val_auc = cv_res['test_roc_auc'].mean()

        overfit_gap = (train_f1 - val_f1) * 100

        if overfit_gap > 15.0:
            risk_label = "🔴 Yüksek Overfit"
        elif overfit_gap > 7.0:
            risk_label = "🟡 Orta Overfit"
        else:
            risk_label = "🟢 Düşük (İyi Genelleme)"

        results.append({
            'Model': model_name,
            'Tuned Train F1': round(train_f1, 4),
            'Tuned CV F1 (Val)': round(val_f1, 4),
            'Tuned Train ROC-AUC': round(train_auc, 4),
            'Tuned CV ROC-AUC (Val)': round(val_auc, 4),
            'Overfit Farkı (%)': round(overfit_gap, 2),
            'Overfit Riski': risk_label,
            'Best Params': params
        })

    df_res = pd.DataFrame(results).sort_values(by='Tuned CV F1 (Val)', ascending=False).reset_index(drop=True)
    return df_res

def calculate_shap_and_bias(model, X_train):

    if isinstance(model, LogisticRegression):
        # Logistic Regression için LinearExplainer
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Ağaç tabanlı modeller (LightGBM, CatBoost, XGBoost, RF, GB) için TreeExplainer
        explainer = shap.TreeExplainer(model)
        
    shap_values = explainer(X_train)

    if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
        shap_vals_target = shap_values.values[:, :, 1]
    elif isinstance(shap_values, list):
        shap_vals_target = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals_target = shap_values.values if hasattr(shap_values, 'values') else shap_values

    mean_abs_shap = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean_SHAP': abs(shap_vals_target).mean(axis=0),
    }).sort_values(by='Mean_SHAP', ascending=False)

    return explainer, shap_values, mean_abs_shap

def detect_anomalies(X_data, contamination=0.05, method='isolation_forest'):
    df_results = X_data.copy()
    if method in ['isolation_forest', 'both']:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        df_results['iso_anomaly'] = iso.fit_predict(X_data)
        df_results['iso_score'] = iso.decision_function(X_data)
        score_col = 'iso_score'
    if method in ['lof', 'both']:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
        df_results['lof_anomaly'] = lof.fit_predict(X_data)
        df_results['lof_score'] = lof.negative_outlier_factor_
        score_col = 'lof_score'

    if method == 'both':
        df_results['is_anomaly'] = np.where((df_results['iso_anomaly'] == -1) & (df_results['lof_anomaly'] == -1), 'Şüpheli / Anomali', 'Normal')
    elif method == 'isolation_forest':
        df_results['is_anomaly'] = np.where(df_results['iso_anomaly'] == -1, 'Şüpheli / Anomali', 'Normal')
    else:
        df_results['is_anomaly'] = np.where(df_results['lof_anomaly'] == -1, 'Şüpheli / Anomali', 'Normal')

    return df_results, score_col

def load_and_prep_test_data(file_path, X_train, target_treatment_col="treatment", target_turnover_col="turnover_risk"):
    """
    Test CSV dosyasını okur, target kolonlarını ayırır ve 
    X_test kolon dizilimini X_train ile birebir hizalar.
    """
    file_path ="dataset/mental_health_test.csv"
    df_test = pd.read_csv(file_path)
    
    # Target'ları çek
    y_test_treatment = df_test[target_treatment_col] if target_treatment_col in df_test.columns else None
    y_test_turnover = df_test[target_turnover_col] if target_turnover_col in df_test.columns else None

    # Feature matrisini oluştur
    cols_to_drop = [col for col in [target_treatment_col, target_turnover_col] if col in df_test.columns]
    X_test = df_test.drop(columns=cols_to_drop)
    
    # Train kolon sırasına ve varlığına eşitle
    X_test = X_test[X_train.columns]

    return df_test, X_test, y_test_treatment, y_test_turnover


def evaluate_model_performance(model, X_test, y_test):
    """
    Verilen model ve test verisine göre tahmin yapar,
    Accuracy ve diğer başarım metriklerini döndürür.
    """
    if model is None or y_test is None:
        return None, None, {}

    # Tahminler
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Başarım / Doğruluk Metrikleri
    metrics = {
        "Accuracy (Doğruluk)": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)

    return y_pred, y_proba, metrics

def predict_sample_employees(file_path, X_train, model_treatment, model_turnover, sample_size=10, random_state=42):
    """
    Test CSV'sinden belirtilen sayıda (varsayılan 10) çalışan örneği seçer.
    Her bir çalışan için Treatment (Destek) ve Turnover (Ayrılma) tahminlerini üretir.
    """
    # 1. Test Verisini Oku
    file_path ="dataset/mental_health_test.csv"

    df_test = pd.read_csv(file_path)
    
    # 2. Rastgele veya İlk 'sample_size' Kadar Çalışanı Seç
    if len(df_test) > sample_size:
        df_sample = df_test.sample(n=sample_size, random_state=random_state).copy()
    else:
        df_sample = df_test.copy()

    # 3. Target Kolonlarını Ayır ve Feature Dizilimini X_train ile Eşitle
    cols_to_drop = [col for col in ["treatment", "turnover_risk"] if col in df_sample.columns]
    X_sample = df_sample.drop(columns=cols_to_drop)
    X_sample = X_sample[X_train.columns]

    # 4. Treatment (Destek Alır mı?) Tahmini
    if model_treatment is not None:
        df_sample["Destek_Tahmini"] = model_treatment.predict(X_sample)
        df_sample["Destek_Alsa_Olasiligi_%"] = (model_treatment.predict_proba(X_sample)[:, 1] * 100).round(1)

    # 5. Turnover (Şirketi Terk Eder mi?) Tahmini
    if model_turnover is not None:
        df_sample["Terk_Tahmini"] = model_turnover.predict(X_sample)
        df_sample["Terk_Etme_Olasiligi_%"] = (model_turnover.predict_proba(X_sample)[:, 1] * 100).round(1)

    # Human-Readable (Okunabilir) Etiketler Ekleyelim
    df_sample["Destek_Durumu"] = df_sample["Destek_Tahmini"].map({1: "✅ Destek Almalı", 0: "❌ İhtiyaç Yok"})
    df_sample["Terk_Durumu"] = df_sample["Terk_Tahmini"].map({1: "🚨 Terk Riski Yüksek", 0: "🟢 Düşük Risk"})

    return df_sample

def fit_selected_model(model_name, best_params_raw, X_train, y_train, random_state=42):
    """
    Optuna'dan gelen parametreleri temizler, seçilen model sınıfını
    örnekler ve train verisi ile fit edip nesneyi döndürür.
    """
    # 1. 'best_params' iç içe (nested) sözlük kontrolü ve kopyalama
    if isinstance(best_params_raw, dict) and "best_params" in best_params_raw:
        model_params = best_params_raw["best_params"].copy()
    elif isinstance(best_params_raw, dict):
        model_params = best_params_raw.copy()
    else:
        model_params = {}

    # 2. Modeli bozabilecek dış metrik anahtarlarını temizle
    model_params.pop("best_cv_f1", None)

    # 3. Model Sınıfı Eşleme ve Örnekleme
    if model_name == "LightGBM":
        model = LGBMClassifier(**model_params, verbose=-1, random_state=random_state)
    elif model_name == "XGBoost":
        model = XGBClassifier(**model_params, random_state=random_state)
    elif model_name == "CatBoost":
        model = CatBoostClassifier(**model_params, verbose=0, random_state=random_state)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(**model_params, random_state=random_state)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(**model_params, random_state=random_state)
    elif model_name == "Logistic Regression":
        # Logistic Regression için konverjans garantisi
        model_params.setdefault("max_iter", 1000)
        model = LogisticRegression(**model_params, random_state=random_state)
    else:
        raise ValueError(f"Geçersiz veya desteklenmeyen model adı: {model_name}")

    # 4. Fit Etme
    model.fit(X_train, y_train)
    
    return model


def fit_both_final_models(X_train, session_state):
    """
    Hem Treatment hem de Turnover Risk modellerini session_state verileriyle fit eder.
    
    Returns:
        final_model_treatment, final_model_turnover
    """
    final_model_treatment = None
    final_model_turnover = None

    # A. Treatment Modeli
    if "selected_treatment_model" in session_state and "best_params_treatment_dict" in session_state:
        chosen_t_name = session_state["selected_treatment_model"]
        raw_t_params = session_state["best_params_treatment_dict"][chosen_t_name]
        y_train_treat = session_state["y_train_treatment"]
        
        final_model_treatment = fit_selected_model(
            model_name=chosen_t_name,
            best_params_raw=raw_t_params,
            X_train=X_train,
            y_train= y_train_treat   
        )

    # B. Turnover Modeli
    if "selected_turnover_model" in session_state and "best_params_turnover_dict" in session_state:
        chosen_turn_name = session_state["selected_turnover_model"]
        raw_turn_params = session_state["best_params_turnover_dict"][chosen_turn_name]
        y_train_turn = session_state["y_train_turnover"]
        
        final_model_turnover = fit_selected_model(
            model_name=chosen_turn_name,
            best_params_raw=raw_turn_params,
            X_train=X_train,
            y_train=y_train_turn
        )
        
   
    return final_model_treatment, final_model_turnover