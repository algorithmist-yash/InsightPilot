import pandas as pd
import numpy as np
from pandas import DataFrame
from utils import log
import config

class CleaningAgent:
    """
    Agent responsible for basic data cleaning operations.
    """
    def __init__(self, threshold: float = 0.6):
        """
        Initializes the agent.
        
        Args:
            threshold (float): Percentage of missing values to flag a column for removal.
        """
        self.threshold = threshold
        log('Cleaning', f'Agent initialized with missing threshold={threshold}.')
        
    def run(self, df: DataFrame) -> (DataFrame, list, list):
        """
        Runs the cleaning process.
        - Identifies high-missing-value columns
        - Standardizes column names
        - Identifies numeric and categorical columns
        
        Args:
            df (DataFrame): The input DataFrame.
            
        Returns:
            tuple: (clean_df, numeric_cols, cat_cols)
                - clean_df (DataFrame): The cleaned DataFrame.
                - numeric_cols (list): List of identified numeric column names.
                - cat_cols (list): List of identified categorical column names.
        """
        log('Cleaning', 'Running cleaning routines...')
        d = df.copy()
        
        # Report and flag columns with > threshold missing
        missing_pct = d.isnull().mean()
        drop_cols = list(missing_pct[missing_pct > self.threshold].index)
        
        if drop_cols:
            log('Cleaning', f'Columns with >{self.threshold*100}% missing (will drop): {drop_cols}')
            d = d.drop(columns=drop_cols)
        else:
            log('Cleaning', 'No columns found with high missing values.')
            
        # Standardize column names (simple strip and lower)
        d.columns = [str(c).strip().lower() for c in d.columns]
        
        # Basic type conversions (example from notebook)
        for c in d.columns:
            if c.lower() == 'survived': # Example from notebook
                d[c] = pd.to_numeric(d[c], errors='coerce')
                
        # Identify final column types
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = d.select_dtypes(include=['object', 'category']).columns.tolist()
        
        log('Cleaning', f'Identified Numeric Cols: {numeric_cols}')
        log('Cleaning', f'Identified Categorical Cols: {cat_cols}')
        
        # Save a preview
        preview_path = config.OUT_DIR / 'clean_preview.csv'
        d.head().to_csv(preview_path, index=False)
        log('Cleaning', f'Clean data preview saved to {preview_path}')
        
        return d, numeric_cols, cat_cols
