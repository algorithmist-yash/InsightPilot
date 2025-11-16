import json
import pandas as pd
from pandas import DataFrame, Series
from utils import log
import config
from pathlib import Path

class ReportingAgent:
    """
    Agent responsible for compiling all results into a final Markdown report.
    """
    def __init__(self):
        log('Reporting', 'Agent initialized.')
        
    def _format_dict(self, data: dict) -> str:
        """Formats a dictionary as a JSON code block."""
        return f"```json\n{json.dumps(data, indent=2)}\n```"

    def _format_df(self, df: DataFrame) -> str:
        """Formats a DataFrame as a Markdown table."""
        if df is None or df.empty:
            return "No data available."
        return df.to_markdown()

    def _format_series(self, series: Series) -> str:
        """Formats a Series as a Markdown table."""
        if series is None or series.empty:
            return "No data available."
        return series.to_frame().to_markdown()

    def run(self, stats: DataFrame, missing: Series, figures: list, model_results: dict) -> str:
        """
        Generates the final Markdown report.
        
        Args:
            stats (DataFrame): Summary statistics from EDAAgent.
            missing (Series): Missing value report from EDAAgent.
            figures (list): List of figure file paths from VisualizationAgent.
            model_results (dict): Dictionary of model results from ModelingAgent.
            
        Returns:
            str: The complete Markdown report as a string.
        """
        log('Reporting', 'Generating final report.md...')
        report_lines = []
        
        report_lines.append("# InsightPilot: Automated EDA & Modeling Report")
        report_lines.append("---")
        
        # --- Summary Stats ---
        report_lines.append("## 1. Summary Statistics")
        report_lines.append("Basic descriptive statistics for all columns.")
        report_lines.append(self._format_df(stats))
        report_lines.append("")

        # --- Missing Values ---
        report_lines.append("## 2. Missing Value Report")
        report_lines.append("Count of missing values per column (descending).")
        report_lines.append(self._format_series(missing))
        report_lines.append("")

        # --- Visualizations ---
        report_lines.append("## 3. Visualizations")
        if not figures:
            report_lines.append("No figures were generated.")
        else:
            report_lines.append("Key plots exploring data distributions and relationships.")
            for fig_path_str in figures:
                fig_path = Path(fig_path_str)
                # Make path relative to the output dir for portability
                relative_path = Path('figures') / fig_path.name
                title = fig_path.stem.replace('_', ' ').title()
                report_lines.append(f"### {title}")
                report_lines.append(f"![{title}]({relative_path})")
                report_lines.append("")
        
        # --- Modeling Results ---
        report_lines.append("## 4. Baseline Model Results")
        if not model_results:
            report_lines.append("No modeling was performed.")
        else:
            report_lines.append("Performance of baseline models on a 20% validation set.")
            report_lines.append(self._format_dict(model_results))
            report_lines.append("")

        report_text = "\n".join(report_lines)
        
        # Save the report
        try:
            report_path = config.OUT_DIR / 'report.md'
            with open(report_path, "w") as f:
                f.write(report_text)
            log('Reporting', f'Report saved to {report_path}')
        except Exception as e:
            log('Reporting', f'Error saving report: {e}')

        return report_text
