from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load



from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


class RealEstatePipeline:

    def __init__(self, n_features_to_select=15, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state

        # stockage modèle final
        self.best_model_ = None
        self.best_model_name_ = None
        self.feature_names_ = None
        self.selected_features_ = None
        self.positive_coef_features_ = None
        self.final_features_ = None
        self.removed_collinear_ = None
        self.dropped_outliers_ = 0
        self.scaler_ = None
        self.imputer_ = None

    # ---------------------------------------------------------
    #         OUTLIER SUPPRESSION (Z-score > 3)
    # ---------------------------------------------------------
    def _remove_outliers(self, df):
        z_scores = np.abs((df - df.mean()) / df.std(ddof=0))
        mask = (z_scores < 3).all(axis=1)
        removed_count = (~mask).sum()
        self.dropped_outliers_ = int(removed_count)
        return df[mask]

    # ---------------------------------------------------------
    #     MULTICOLINEARITÉ ─ correlation > 0.90
    # ---------------------------------------------------------
    def _remove_multicollinearity(self, df, target_name):
        corr_matrix = df.drop(columns=[target_name]).corr().abs()
        to_remove = set()

        target_corr = df.corr()[target_name].abs()

        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and corr_matrix.loc[col1, col2] > 0.90:
                    # garde la plus corrélée au target
                    if target_corr[col1] < target_corr[col2]:
                        to_remove.add(col1)
                    else:
                        to_remove.add(col2)

        self.removed_collinear_ = list(to_remove)
        return df.drop(columns=to_remove)

    # ---------------------------------------------------------
    #                 FIT PIPELINE COMPLET
    # ---------------------------------------------------------
    def fit(self, df, target_col="prix_fcfa", cv=3, scoring="neg_root_mean_squared_error", verbose=1):
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in df")

        # 1) Remove outliers
        df = self._remove_outliers(df)

        # 2) Remove multicollinearity
        df = self._remove_multicollinearity(df, target_col)

        # 3) Split
        y = df[target_col]
        X = df.drop(columns=[target_col])
        self.feature_names_ = X.columns.tolist()

        # 4) Imputer + Scaler
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        # 5) RFE pour sélection de variables
        rfe_base = LinearRegression()
        rfe = RFE(estimator=rfe_base, n_features_to_select=self.n_features_to_select)
        rfe.fit(X_scaled, y)
        rfe_features = [f for f, keep in zip(self.feature_names_, rfe.support_) if keep]
        self.selected_features_ = rfe_features

        # 6) Garder que les coefficients positifs
        lin = LinearRegression().fit(X_scaled[:, rfe.support_], y)
        coefs = lin.coef_
        positive_mask = coefs > 0

        self.positive_coef_features_ = [f for f, keep in zip(rfe_features, positive_mask) if keep]

        # final feature set
        self.final_features_ = self.positive_coef_features_
        X_final = X[self.final_features_]

        # 7) Impute + scale final
        Xf_imp = self.imputer_.fit_transform(X_final)
        Xf_scaled = self.scaler_.fit_transform(Xf_imp)

        # 8) Grid Search on linear models
        models = {
            "LinearRegression": (LinearRegression(), {}),
            "Lasso": (Lasso(max_iter=5000), {"alpha": [0.001, 0.01, 0.1, 1]}),
            "Ridge": (Ridge(), {"alpha": [0.1, 1, 10]}),
            "ElasticNet": (ElasticNet(max_iter=5000),
                           {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]})
        }

        best_score = -np.inf
        best_model = None
        best_name = None

        for name, (estimator, params) in models.items():
            gs = GridSearchCV(estimator, params, cv=cv, scoring=scoring, verbose=verbose)
            gs.fit(Xf_scaled, y)

            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_model = gs.best_estimator_
                best_name = name

        self.best_model_ = best_model
        self.best_model_name_ = best_name

        return self

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    def predict(self, X_new):
        X_new = pd.DataFrame([X_new]) if isinstance(X_new, dict) else X_new.copy()

        missing_cols = set(self.final_features_) - set(X_new.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        X_new = X_new[self.final_features_]

        X_imp = self.imputer_.transform(X_new)
        X_scaled = self.scaler_.transform(X_imp)

        return self.best_model_.predict(X_scaled)

    # ---------------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------------
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------
    def diagnostics(self):
        coef = self.best_model_.coef_
        return pd.DataFrame({
            "feature": self.final_features_,
            "coefficient": coef,
            "impact": ["positive" if c > 0 else "negative" for c in coef]
        }).sort_values(by="coefficient", ascending=False)




# Charger le modèle
pipeline = load("pipe.joblib")

app = FastAPI(title="Predicteur du prix de villa en vente")

# Définition des entrées
class InputData(BaseModel):
    superficie_m2: int
    nombre_pieces: int
    nombre_salles_bain: int
    jardin:int
    piscine: int
    parking: int
    cuisineEquipee: int
    securisee: int
    standing: int
    cite: int
    magasin: int
    acces: int
    meuble: int
    non_finition: int
    basse: int
    duplex: int
    triplex: int
    prix_moyen: float
    prix_min: float
    prix_max: float
    prix_median: float
    prix_q1: float
    prix_q3: float
    variance_prix: float
    titreFoncier :int

@app.post("/")
def predict_price(data: InputData):
    # Convertir en DataFrame avec une seule ligne
    df = pd.DataFrame([data.dict()])

    # Prédiction
    pred = pipeline.predict(df)[0]

    return {"prediction": float(pred)}
