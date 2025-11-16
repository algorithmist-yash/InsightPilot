import pandas as pd
from pandas import DataFrame
from utils import log
import config

class EDAAgent:
    """
    Agent responsible for computing descriptive statistics and missing value reports.
    """
    def __init__(self):
        log('EDA', 'Agent initialized.')
        
    def run(self, df: DataFrame) -> (pd.DataFrame, pd.Series):
        """
        Generates summary statistics and a missing value report.
        
        Args:
            df (DataFrame): The (cleaned) input DataFrame.
            
        Returns:
            tuple: (summary_stats, missing_report)
                - summary_stats (DataFrame): describe() output.
                - missing_report (Series): isnull().sum() output.
        """
        log('EDA', 'Computing summary statistics and missing value report...')
        
        # Summary statistics
        try:
            summary = df.describe(include='all').T
            summary_path = config.OUT_DIR / 'summary_stats.csv'
            summary.to_csv(summary_path)
            log('EDA', f'Summary stats saved to {summary_path}')
        except Exception as e:
            log('EDA', f'Error generating summary stats: {e}')
            summary = pd.DataFrame() # Return empty
            
        # Missing value report
        try:
            missing_report = df.isnull().sum().sort_values(ascending=False)
            missing_path = config.OUT_DIR / 'missing_report.csv'
            missing_report.to_csv(missing_path)
            log('EDA', f'Missing value report saved to {missing_path}')
        except Exception as e:
            log('EDA', f'Error generating missing report: {e}')
            missing_report = pd.Series() # Return empty

        return summary, missing_report
