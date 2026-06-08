"""
Custom problem management for user-defined MCDM problems
"""

import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional

# Session state key under which a student's custom problems are stored.
# Storing per-session (instead of a shared folder on disk) keeps every
# student's custom problems isolated so concurrent users don't see,
# overwrite, or delete each other's problems.
CUSTOM_PROBLEMS_KEY = "custom_problems_store"

def _get_store() -> Dict[str, Dict]:
    """Return this session's custom problems store, creating it if needed."""
    if CUSTOM_PROBLEMS_KEY not in st.session_state:
        st.session_state[CUSTOM_PROBLEMS_KEY] = {}
    return st.session_state[CUSTOM_PROBLEMS_KEY]

def save_custom_problem(name: str, description: str, alternatives: List[str], 
                       criteria: List[str], criterion_types: List[str], 
                       decision_matrix: pd.DataFrame, weights: List[float]) -> bool:
    """
    Save a custom problem to the current session
    
    Args:
        name: Problem name
        description: Problem description
        alternatives: List of alternative names
        criteria: List of criteria names
        criterion_types: List of criterion types (benefit/cost)
        decision_matrix: Decision matrix DataFrame
        weights: List of criteria weights
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Create problem data structure
        problem_data = {
            "name": name,
            "description": description,
            "alternatives": alternatives,
            "criteria": criteria,
            "criterion_types": criterion_types,
            "decision_matrix": decision_matrix.values.tolist() if decision_matrix is not None else None,
            "weights": weights,
            "created_date": datetime.now().isoformat(),
            "custom": True
        }
        
        # Store in this session only
        _get_store()[name] = problem_data
        
        return True
        
    except Exception as e:
        st.error(f"Error saving custom problem: {str(e)}")
        return False

def load_custom_problems() -> Dict[str, Dict]:
    """
    Load all custom problems for the current session
    
    Returns:
        Dict: Dictionary of custom problems
    """
    # Return a shallow copy so callers can't mutate the store directly
    return dict(_get_store())

def delete_custom_problem(name: str) -> bool:
    """
    Delete a custom problem from the current session
    
    Args:
        name: Problem name to delete
        
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    try:
        store = _get_store()
        if name in store:
            del store[name]
            return True
        else:
            st.warning(f"Custom problem '{name}' not found")
            return False
            
    except Exception as e:
        st.error(f"Error deleting custom problem: {str(e)}")
        return False

def export_custom_problem(name: str) -> Optional[str]:
    """
    Export a custom problem as JSON string
    
    Args:
        name: Problem name to export
        
    Returns:
        str: JSON string of the problem, or None if error
    """
    try:
        store = _get_store()
        if name in store:
            return json.dumps(store[name], indent=2)
        else:
            st.warning(f"Custom problem '{name}' not found")
            return None
            
    except Exception as e:
        st.error(f"Error exporting custom problem: {str(e)}")
        return None

def import_custom_problem(json_data: str) -> bool:
    """
    Import a custom problem from JSON string
    
    Args:
        json_data: JSON string containing problem data
        
    Returns:
        bool: True if imported successfully, False otherwise
    """
    try:
        problem_data = json.loads(json_data)
        
        # Validate required fields
        required_fields = ['name', 'description', 'alternatives', 'criteria', 'criterion_types']
        for field in required_fields:
            if field not in problem_data:
                st.error(f"Missing required field: {field}")
                return False
        
        # Save the imported problem
        name = problem_data['name']
        description = problem_data['description']
        alternatives = problem_data['alternatives']
        criteria = problem_data['criteria']
        criterion_types = problem_data['criterion_types']
        weights = problem_data.get('weights', [1.0/len(criteria)] * len(criteria))
        
        # Convert decision matrix back to DataFrame
        decision_matrix = None
        if problem_data.get('decision_matrix'):
            decision_matrix = pd.DataFrame(
                problem_data['decision_matrix'],
                index=alternatives,
                columns=criteria
            )
        
        return save_custom_problem(name, description, alternatives, criteria, 
                                 criterion_types, decision_matrix, weights)
        
    except json.JSONDecodeError:
        st.error("Invalid JSON format")
        return False
    except Exception as e:
        st.error(f"Error importing custom problem: {str(e)}")
        return False

def get_all_problems():
    """
    Get all problems (built-in + custom)
    
    Returns:
        Dict: Combined dictionary of all problems
    """
    from config.settings import EXAMPLE_PROBLEMS
    
    # Start with built-in problems
    all_problems = EXAMPLE_PROBLEMS.copy()
    
    # Add custom problems (per-session)
    custom_problems = load_custom_problems()
    for name, problem in custom_problems.items():
        # Mark as custom and add to all problems
        problem['custom'] = True
        all_problems[name] = problem
    
    return all_problems

def validate_problem_data(alternatives: List[str], criteria: List[str], 
                         criterion_types: List[str], decision_matrix: pd.DataFrame, 
                         weights: List[float]) -> List[str]:
    """
    Validate problem data for consistency
    
    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []
    
    # Check alternatives
    if len(alternatives) < 2:
        errors.append("At least 2 alternatives are required")
    
    if len(set(alternatives)) != len(alternatives):
        errors.append("Alternative names must be unique")
    
    # Check criteria
    if len(criteria) < 1:
        errors.append("At least 1 criterion is required")
    
    if len(set(criteria)) != len(criteria):
        errors.append("Criteria names must be unique")
    
    # Check criterion types
    if len(criterion_types) != len(criteria):
        errors.append("Number of criterion types must match number of criteria")
    
    valid_types = ['benefit', 'cost']
    for i, ctype in enumerate(criterion_types):
        if ctype not in valid_types:
            errors.append(f"Invalid criterion type '{ctype}' for criterion {i+1}")
    
    # Check decision matrix
    if decision_matrix is not None:
        if decision_matrix.shape != (len(alternatives), len(criteria)):
            errors.append("Decision matrix dimensions don't match alternatives and criteria")
        
        if decision_matrix.isnull().any().any():
            errors.append("Decision matrix contains missing values")
    
    # Check weights
    if len(weights) != len(criteria):
        errors.append("Number of weights must match number of criteria")
    
    if any(w < 0 for w in weights):
        errors.append("Weights cannot be negative")
    
    if sum(weights) == 0:
        errors.append("Sum of weights cannot be zero")
    
    return errors
