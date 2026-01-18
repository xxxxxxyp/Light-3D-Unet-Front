"""
Configuration Manager Module
Handles loading and saving of YAML configurations.
"""

import os
import yaml

class ConfigManager:
    """Helper class to load and save configuration files"""
    
    @staticmethod
    def load(path):
        """Load configuration from a YAML file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save(config, path):
        """Save configuration to a YAML file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, "w", encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)