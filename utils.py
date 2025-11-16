import json
import numpy as np
import pandas as pd

# Global log to store agent messages
AGENT_LOG = []

def log(agent: str, msg: str):
    """
    Logs a message from an agent and prints it.
    
    Args:
        agent (str): The name of the agent logging the message.
        msg (str): The message to log.
    """
    message_entry = {'agent': agent, 'message': str(msg)}
    AGENT_LOG.append(message_entry)
    print(f"[{agent}] {msg}")

def dataset_metadata(df: pd.DataFrame) -> dict:
    """
    Inspects a DataFrame and returns key metadata.
    
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        
    Returns:
        dict: A dictionary containing metadata (rows, cols, dtypes, missing_values).
    """
    return {
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }

def save_log(log_path: str):
    """
    Saves the global AGENT_LOG to a JSON file.
    
    Args:
        log_path (str): The file path to save the JSON log.
    """
    try:
        with open(log_path, 'w') as f:
            json.dump(AGENT_LOG, f, indent=2)
        log('Utils', f"Agent log saved to {log_path}")
    except Exception as e:
        log('Utils', f"Error saving agent log: {e}")
