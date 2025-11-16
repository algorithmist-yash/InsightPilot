import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from utils import log
import config

class VisualizationAgent:
    """
    Agent responsible for generating and saving visualizations.
    """
    def __init__(self):
        log('Viz', 'Agent initialized.')
        plt.rcParams['figure.max_open_warning'] = 50 # Suppress warning
        sns.set_theme(style="whitegrid") # Set a nice default theme

    def _savefig(self, fig, name: str) -> str:
        """
        Helper to save a figure and close it.
        
        Args:
            fig (matplotlib.figure.Figure): The figure object.
            name (str): The filename (e.g., 'plot.png').
            
        Returns:
            str: The full path to the saved figure.
        """
        path = config.FIG_DIR / name
        try:
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            log('Viz', f'Saved {path}')
            return str(path)
        except Exception as e:
            log('Viz', f'Error saving figure {name}: {e}')
            plt.close(fig)
            return ""

    def run(self, df: DataFrame, numeric_cols: list, cat_cols: list) -> list:
        """
        Generates a standard set of plots.
        
        Args:
            df (DataFrame): The (cleaned) input DataFrame.
            numeric_cols (list): List of numeric column names.
            cat_cols (list): List of categorical column names.
            
        Returns:
            list: A list of file paths for the generated figures.
        """
        log('Viz', 'Generating visualizations...')
        figure_paths = []

        # 1. Correlation heatmap for numeric features
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', ax=ax, cmap='coolwarm')
                ax.set_title('Numeric Correlation Matrix')
                path = self._savefig(fig, 'correlation_matrix.png')
                if path: figure_paths.append(path)
            except Exception as e:
                log('Viz', f'Failed to generate correlation matrix: {e}')

        # 2. Histograms for numeric features
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots()
                df[col].dropna().plot.hist(bins=30, ax=ax, edgecolor='k')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                path = self._savefig(fig, f'hist_{col}.png')
                if path: figure_paths.append(path)
            except Exception as e:
                log('Viz', f'Failed to generate histogram for {col}: {e}')

        # 3. Count plots for low-cardinality categorical features
        for col in cat_cols:
            try:
                n_unique = df[col].nunique()
                if n_unique > 1 and n_unique < 50: # Avoid plotting high-cardinality
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, y=col, ax=ax, order=df[col].value_counts().index)
                    ax.set_title(f'Count of {col}')
                    ax.set_xlabel('Count')
                    path = self._savefig(fig, f'count_{col}.png')
                    if path: figure_paths.append(path)
                else:
                    log('Viz', f'Skipping count plot for {col} (unique values: {n_unique})')
            except Exception as e:
                log('Viz', f'Failed to generate count plot for {col}: {e}')
        
        # --- Domain-specific plots (Example from your notebook) ---
        if set(['survived','pclass']).issubset(df.columns):
            try:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='pclass', hue='survived', ax=ax)
                ax.set_title('Survival by Passenger Class')
                path = self._savefig(fig, 'domain_survival_by_pclass.png')
                if path: figure_paths.append(path)
            except Exception as e:
                log('Viz', f'Failed to generate domain plot "Survival by Pclass": {e}')

        log('Viz', 'Visualization step complete.')
        return figure_paths
