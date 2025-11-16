import json
from pandas import DataFrame
from utils import log, dataset_metadata

class PlannerAgent:
    """
    Agent responsible for inspecting the dataset and creating a workflow plan.
    """
    def __init__(self):
        log('Planner', 'Agent initialized.')
    
    def run(self, df: DataFrame) -> (list, dict):
        """
        Inspects the dataset and decides which tasks to run.
        
        Args:
            df (DataFrame): The input DataFrame.
            
        Returns:
            tuple: (plan, metadata)
                - plan (list): A list of agent names (tasks) to run.
                - metadata (dict): A dictionary of metadata about the dataset.
        """
        log('Planner', 'Inspecting dataset to decide workflow...')
        
        metadata = dataset_metadata(df)
        log('Planner', json.dumps({'rows': metadata['rows'], 'cols': metadata['cols']}))
        
        # --- Plan Decision Logic ---
        # For this project, the plan is static, but this is where
        # you could add logic (e.g., skip modeling if no target is found).
        plan = ['cleaning', 'eda', 'visualization']
        
        # Add modeling to plan only if a plausible target column exists
        if any(col.lower() in [c.lower() for c in df.columns] for col in ['target', 'survived', 'class', 'label']):
             plan.append('modeling')
        else:
            log('Planner', 'No obvious target column found, skipping modeling.')

        plan.append('reporting')
        
        log('Planner', f'Generated plan: {plan}')
        return plan, metadata
