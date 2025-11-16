import json
import numpy as np
from pandas import DataFrame
from utils import log
import config

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

class ModelingAgent:
    """
    Agent responsible for training and evaluating baseline models.
    """
    def __init__(self, target_candidates: list, random_seed: int):
        self.target_candidates = [t.lower() for t in target_candidates]
        self.random_seed = random_seed
        self.target_col = None
        log('Modeling', 'Agent initialized.')
        
    def _find_target(self, df: DataFrame) -> str:
        """Finds the first matching target column from the candidates."""
        df_cols_lower = [c.lower() for c in df.columns]
        for candidate in self.target_candidates:
            if candidate in df_cols_lower:
                # Get the original case column name
                original_col = df.columns[df_cols_lower.index(candidate)]
                log('Modeling', f'Detected target column: {original_col}')
                return original_col
        log('Modeling', 'No target column detected from candidates.')
        return None

    def run(self, df: DataFrame) -> dict:
        """
        Runs the baseline modeling pipeline.
        
        Args:
            df (DataFrame): The (cleaned) input DataFrame.
            
        Returns:
            dict: A dictionary of model results.
        """
        log('Modeling', 'Attempting baseline supervised modeling...')
        
        self.target_col = self._find_target(df)
        
        if self.target_col is None:
            log('Modeling', 'Skipping modeling.')
            return {}

        try:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            # Ensure target is not all one class (which breaks ROC-AUC)
            if y.nunique() < 2:
                log('Modeling', f'Target column "{self.target_col}" has only one class. Skipping modeling.')
                return {}

            # Identify features for preprocessing
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
            # Filter categorical columns to those with reasonable cardinality
            cat_cols = [c for c in cat_cols if X[c].nunique() < 50]
            
            log('Modeling', f'Using {len(num_cols)} numeric features and {len(cat_cols)} categorical features.')

            # Build preprocessing pipelines
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                [('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)],
                remainder='drop'
            )

            # Define models
            models = {
                'logreg': LogisticRegression(max_iter=1000, random_state=self.random_seed),
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
            }
            
            # Filter models based on config
            models_to_run = {k: v for k, v in models.items() if k in config.MODELS_TO_RUN}
            if not models_to_run:
                log('Modeling', 'No models specified in config.MODELS_TO_RUN. Skipping.')
                return {}

            # Train/test split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed, stratify=y
            )

            model_results = {}
            for name, model in models_to_run.items():
                log('Modeling', f'Training {name}...')
                # Create full pipeline
                pipe = Pipeline([('prep', preprocessor), ('clf', model)])
                
                try:
                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_val)
                    
                    # Get probabilities for ROC-AUC
                    prob = None
                    if hasattr(pipe, 'predict_proba'):
                        prob = pipe.predict_proba(X_val)[:, 1]
                        
                    acc = accuracy_score(y_val, preds)
                    roc = roc_auc_score(y_val, prob) if prob is not None else None
                    
                    report = classification_report(y_val, preds, output_dict=True)
                    
                    model_results[name] = {
                        'accuracy': float(acc), 
                        'roc_auc': (float(roc) if roc is not None else 'N/A'),
                        'classification_report': report
                    }
                    log('Modeling', f'{name}: accuracy={acc:.4f} roc_auc={roc if roc is not None else "N/A"}')

                except Exception as e:
                    log('Modeling', f'Training/evaluation for {name} FAILED: {e}')
                    model_results[name] = {'error': str(e)}

            # Save results
            results_path = config.OUT_DIR / 'model_results.json'
            with open(results_path, 'w') as f:
                json.dump(model_results, f, indent=2)
            log('Modeling', f'Model results saved to {results_path}')
            
            return model_results

        except Exception as e:
            log('Modeling', f'Major modeling failure: {e}')
            return {'error': str(e)}
