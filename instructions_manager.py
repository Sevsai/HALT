import os
import json
import logging

class InstructionsManager:
    """Manages system instructions for the application"""
    
    def __init__(self, file_path=None):
        """Initialize the instructions manager with the given file path"""
        self.file_path = file_path
        self.instructions = {
            "current_system_instruction": "General Assistant",
            "system_instructions": {
                "General Assistant": "You are a helpful, respectful assistant. Provide accurate, detailed responses.",
                "Code Assistant": "You are a programming assistant. Help with writing, debugging, and explaining code.",
                "Creative Writer": "You are a creative writing assistant. Help with generating creative content and ideas."
            }
        }
        
        # Load instructions if file exists
        if file_path and os.path.exists(file_path):
            self.load_instructions()
        
    def load_instructions(self):
        """Load instructions from file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.instructions = json.load(f)
            logging.info(f"Loaded instructions from {self.file_path}")
        except Exception as e:
            logging.error(f"Failed to load instructions: {e}")
            
    def save_instructions(self):
        """Save instructions to file"""
        if not self.file_path:
            logging.error("Cannot save instructions: No file path specified")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            # Save to file
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.instructions, f, indent=2)
            logging.info(f"Saved instructions to {self.file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save instructions: {e}")
            return False
            
    def get_system_instruction_names(self):
        """Get list of available system instruction names"""
        return list(self.instructions.get("system_instructions", {}).keys())
            
    def get_system_instruction(self, name):
        """Get a specific system instruction by name"""
        return self.instructions.get("system_instructions", {}).get(name, "")
            
    def get_current_system_instruction(self):
        """Get the current system instruction"""
        current_name = self.instructions.get("current_system_instruction", "")
        return self.get_system_instruction(current_name)
            
    def set_system_instruction(self, name, instruction_text):
        """Set or update a system instruction"""
        if "system_instructions" not in self.instructions:
            self.instructions["system_instructions"] = {}
            
        self.instructions["system_instructions"][name] = instruction_text
        return self.save_instructions()
            
    def set_current_system_instruction(self, name):
        """Set the current system instruction"""
        if name in self.instructions.get("system_instructions", {}):
            self.instructions["current_system_instruction"] = name
            return self.save_instructions()
        return False
