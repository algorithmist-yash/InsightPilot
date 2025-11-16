import os
import zipfile
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import project configurations and utilities
import config
from utils import log, save_log

# Import all agents
from agents.planner_agent import PlannerAgent
from agents.cleaning_agent import CleaningAgent
from agents.eda_agent import EDAAgent
from agents.viz_agent import VisualizationAgent
from agents.modeling_agent import ModelingAgent
from agents.reporting_agent import ReportingAgent

def setup_directories():
    """
    Creates the output directories defined in the config.
    """
    log('Main', f"Setting up output directory: {config.OUT_DIR}")
    config.OUT_DIR.mkdir(exist_ok=True)
    config.FIG_DIR.mkdir(exist_ok=True)

def load_data() -> pd.DataFrame:
    """
    Loads the primary dataset.
    It searches for 'train.csv' or a zip file containing it.
    If not found, it falls back to a toy dataset.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    log('DataLoader', 'Searching for dataset...')
    
    # Standard paths
    train_path_csv = Path('train.csv')
    train_path_zip = Path('titanic.zip') # Common name on Kaggle
    kaggle_input_path = Path('/kaggle/input/titanic/train.csv') # Kaggle env path

    train_path = None

    if kaggle_input_path.exists():
        train_path = kaggle_input_path
    elif train_path_csv.exists():
        train_path = train_path_csv
    elif train_path_zip.exists():
        log('DataLoader', f"Found {train_path_zip}, attempting to extract 'train.csv'")
        try:
            with zipfile.ZipFile(train_path_zip, 'r') as z:
                # Find 'train.csv' within the zip
                for name in z.namelist():
                    if name.endswith('train.csv'):
                        z.extract(name, path='.')
                        train_path = Path(name)
                        log('DataLoader', f"Extracted {train_path}")
                        break
        except Exception as e:
            log('DataLoader', f"Failed to extract from zip: {e}")
            train_path = None # Ensure fallback

    if train_path is None or not train_path.exists():
        log('DataLoader', 'No dataset found. Loading fallback toy dataset for demonstration.')
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, np.nan],
            'B': [5, 4, 3, 2, 1, 5],
            'C': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
            'target': [0, 1, 0, 1, 0, 1]
        })
    else:
        try:
            df = pd.read_csv(train_path)
            log('DataLoader', f"Successfully loaded {train_path} with shape {df.shape}")
        except Exception as e:
            log('DataLoader', f"Error reading {train_path}: {e}. Using fallback.")
            df = pd.DataFrame({'target': [0,1]}) # Minimal fallback

    log('DataLoader', 'Preview of loaded data:')
    print(df.head())
    return df

def run_pipeline():
    """
    Main orchestrator for the multi-agent pipeline.
    """
    log('Main', '--- InsightPilot Pipeline Started ---')
    
    # 1. Setup
    setup_directories()
    df = load_data()
    
    # Store results from each agent
    pipeline_memory = {
        'original_df': df
    }

    # 2. Initialize Agents
    planner = PlannerAgent()
    cleaner = CleaningAgent(threshold=config.CLEANING_MISSING_THRESHOLD)
    eda = EDAAgent()
    visualizer = VisualizationAgent()
    modeler = ModelingAgent(
        target_candidates=config.COMMON_TARGET_NAMES, 
        random_seed=config.RANDOM_SEED
    )
    reporter = ReportingAgent()

    # 3. Run Agent Pipeline
    try:
        # --- Planner ---
        plan, metadata = planner.run(df)
        pipeline_memory['metadata'] = metadata
        log('Main', f"Plan generated: {plan}")

        # --- Cleaner ---
        if 'cleaning' in plan:
            clean_df, num_cols, cat_cols = cleaner.run(df)
            pipeline_memory['clean_df'] = clean_df
            pipeline_memory['numeric_cols'] = num_cols
            pipeline_memory['categorical_cols'] = cat_cols
        else:
            log('Main', 'Skipping cleaning.')
            pipeline_memory['clean_df'] = df
            pipeline_memory['numeric_cols'] = df.select_dtypes(include=[np.number]).columns.tolist()
            pipeline_memory['categorical_cols'] = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # --- EDA ---
        if 'eda' in plan:
            stats, missing = eda.run(pipeline_memory['clean_df'])
            pipeline_memory['summary_stats'] = stats
            pipeline_memory['missing_report'] = missing

        # --- Visualization ---
        if 'visualization' in plan:
            fig_paths = visualizer.run(
                pipeline_memory['clean_df'], 
                pipeline_memory['numeric_cols'],
                pipeline_memory['categorical_cols']
            )
            pipeline_memory['figure_paths'] = fig_paths

        # --- Modeling ---
        if 'modeling' in plan:
            model_results = modeler.run(pipeline_memory['clean_df'])
            pipeline_memory['model_results'] = model_results

        # --- Reporting ---
        if 'reporting' in plan:
            report_str = reporter.run(
                pipeline_memory.get('summary_stats'),
                pipeline_memory.get('missing_report'),
                pipeline_memory.get('figure_paths', []),
                pipeline_memory.get('model_results', {})
            )
            pipeline_memory['final_report'] = report_str
            log('Main', 'Final report generated.')

    except Exception as e:
        log('Main', f'FATAL ERROR in pipeline: {e}')
    
    # 4. Save agent log
    save_log(config.OUT_DIR / 'agent_log.json')
    log('Main', '--- InsightPilot Pipeline Finished ---')

if __name__ == "__main__":
    # Set numpy random seed from config
    np.random.seed(config.RANDOM_SEED)
    run_pipeline()
