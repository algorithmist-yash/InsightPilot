from pathlib import Path

# Seed for reproducibility
RANDOM_SEED = 42

# --- Directory Settings ---

# Main output directory
OUT_DIR = Path('output')

# Subdirectory for generated figures
FIG_DIR = OUT_DIR / 'figures'

# --- Agent Settings ---

# Columns with a missing percentage higher than this will be flagged for dropping
CLEANING_MISSING_THRESHOLD = 0.6

# Common names for a target variable
COMMON_TARGET_NAMES = ['Survived', 'target', 'label', 'class']

# Models to run in the modeling agent
MODELS_TO_RUN = {
    'logreg': 'LogisticRegression',
    'rf': 'RandomForestClassifier'
}
