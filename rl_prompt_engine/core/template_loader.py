#!/usr/bin/env python3
"""
Template Loader for RL Prompt Engine

Loads and processes markdown templates for prompt generation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class TemplateLoader:
    """Loads and processes markdown templates."""
    
    def __init__(self, template_dir: str = "rl_prompt_engine/templates"):
        """
        Initialize template loader.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = Path(template_dir)
        
    def load_template(self, template_name: str) -> str:
        """
        Load a template file.
        
        Args:
            template_name: Name of template file (with or without .md extension)
            
        Returns:
            Template content as string
            
        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if not template_name.endswith('.md'):
            template_name += '.md'
            
        template_path = self.template_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
            
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def render_template(self, 
                      template_name: str, 
                      variables: Dict[str, Any]) -> str:
        """
        Load and render a template with variables.
        
        Args:
            template_name: Name of template file
            variables: Dictionary of variables to substitute
            
        Returns:
            Rendered template string
        """
        template = self.load_template(template_name)
        
        # Replace variables in template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            template = template.replace(placeholder, str(value))
            
        return template
    
    def get_available_templates(self) -> list:
        """
        Get list of available template files.
        
        Returns:
            List of template file names (without .md extension)
        """
        if not self.template_dir.exists():
            return []
            
        templates = []
        for file_path in self.template_dir.glob("*.md"):
            templates.append(file_path.stem)
            
        return sorted(templates)
    
    def validate_template(self, template_name: str) -> Dict[str, list]:
        """
        Validate a template and return missing variables.
        
        Args:
            template_name: Name of template file
            
        Returns:
            Dictionary with 'missing_variables' and 'found_variables' lists
        """
        template = self.load_template(template_name)
        
        # Find all {variable} patterns
        import re
        pattern = r'\{([^}]+)\}'
        found_variables = re.findall(pattern, template)
        
        # Remove duplicates and sort
        found_variables = sorted(list(set(found_variables)))
        
        return {
            'found_variables': found_variables,
            'template_content': template
        }
