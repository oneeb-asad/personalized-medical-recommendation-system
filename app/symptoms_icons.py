"""
Symptom Icons Mapping
This module provides icon mapping for different symptoms in the Medical Recommendation System
"""

def get_symptom_icon(symptom):
    """
    Returns the appropriate FontAwesome icon class for a given symptom.
    
    Args:
        symptom (str): The symptom name
        
    Returns:
        str: FontAwesome icon class name
    """
    symptom_lower = symptom.lower()
    
    # Comprehensive symptom-to-icon mapping
    icon_mapping = {
        # Respiratory symptoms
        'cough': 'fas fa-lungs',
        'breath': 'fas fa-lungs-virus',
        'sneez': 'fas fa-head-side-cough',
        'throat': 'fas fa-head-side-cough-slash',
        'phlegm': 'fas fa-lungs',
        
        # Temperature and fever
        'fever': 'fas fa-thermometer-full',
        'chills': 'fas fa-snowflake',
        
        # Pain symptoms
        'pain': 'fas fa-band-aid',
        'headache': 'fas fa-head-side-virus',
        'joint': 'fas fa-bone',
        'muscle': 'fas fa-dumbbell',
        'burning': 'fas fa-fire',
        
        # Digestive symptoms
        'nausea': 'fas fa-dizzy',
        'vomit': 'fas fa-dizzy',
        'stomach': 'fas fa-procedures',
        'diarr': 'fas fa-toilet',
        'constipat': 'fas fa-ban',
        'acidity': 'fas fa-burn',
        'indigestion': 'fas fa-stomach',
        'ulcer': 'fas fa-circle-notch',
        
        # General symptoms
        'fatigue': 'fas fa-bed',
        'lethargy': 'fas fa-battery-quarter',
        'restless': 'fas fa-running',
        'malaise': 'fas fa-frown',
        
        # Skin symptoms
        'rash': 'fas fa-allergies',
        'skin': 'fas fa-allergies',
        'itch': 'fas fa-hand-paper',
        'spots': 'fas fa-circle',
        'patches': 'fas fa-circle',
        'yellow': 'fas fa-exclamation-triangle',
        
        # Body functions
        'sweat': 'fas fa-tint',
        'urine': 'fas fa-flask',
        'weight': 'fas fa-weight',
        'swelling': 'fas fa-plus-circle',
        'dehydrat': 'fas fa-tint-slash',
        
        # Mental/Neurological
        'mood': 'fas fa-brain',
        'anxiety': 'fas fa-heart',
        'sunken': 'fas fa-eye-slash',
        'blurred': 'fas fa-glasses',
        'vision': 'fas fa-glasses',
        
        # Body systems
        'eye': 'fas fa-eye',
        'sugar': 'fas fa-candy-cane',
        'nodal': 'fas fa-dot-circle',
        'lymph': 'fas fa-dot-circle',
        'liver': 'fas fa-prescription-bottle',
        'micturition': 'fas fa-fire-alt',
        
        # Specific conditions
        'appetite': 'fas fa-utensils-slash',
        'fluid': 'fas fa-fill-drip',
        'overload': 'fas fa-fill-drip',
        'dark': 'fas fa-vial',
    }
    
    # Check for keyword matches
    for keyword, icon in icon_mapping.items():
        if keyword in symptom_lower:
            # Special cases for compound conditions
            if keyword == 'loss' and 'appetite' in symptom_lower:
                return 'fas fa-utensils-slash'
            elif keyword == 'fluid' and 'overload' in symptom_lower:
                return 'fas fa-fill-drip'
            elif keyword == 'dark' and 'urine' in symptom_lower:
                return 'fas fa-vial'
            else:
                return icon
    
    # Default icon if no match found
    return 'fas fa-stethoscope'


def get_all_symptom_icons(symptoms_list):
    """
    Returns a dictionary mapping each symptom to its icon.
    
    Args:
        symptoms_list (list): List of symptom names
        
    Returns:
        dict: Dictionary with symptom names as keys and icon classes as values
    """
    return {symptom: get_symptom_icon(symptom) for symptom in symptoms_list}