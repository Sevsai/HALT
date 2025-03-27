#!/usr/bin/env python
import os
import sys
import time
import json
import logging
import threading
from threading import Thread, Event
import asyncio

# Add parent directory to Python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from instructions_manager import InstructionsManager

from PyQt5.QtWidgets import (QApplication, QMainWindow, QSplitter, QTabWidget, 
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTextEdit, QFrame, QCheckBox, QComboBox, QSpinBox, 
                             QScrollArea, QLineEdit, QFileDialog, QMessageBox, QColorDialog, QListWidget,
                             QShortcut, QToolButton, QInputDialog, QDialog, QFontDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer, QObject, QMimeData, QByteArray, QMetaObject, Q_ARG
from PyQt5.QtGui import QFont, QColor, QPixmap, QTextCursor, QIcon, QDrag, QKeySequence
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

from dotenv import load_dotenv

# Import the instructions manager
from instructions_manager import InstructionsManager

# LocalModelHandler is defined in this file
# Removing the import since it's defined within this file

# ------------------------------------------------------------------------------
# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import logging

# Add GPU utilities after imports
def get_available_gpus():
    """Get list of available NVIDIA GPUs"""
    available_gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            available_gpus.append((i, gpu_name))
    return available_gpus

def optimize_gpu_settings():
    """Apply optimizations for CUDA performance"""
    if not torch.cuda.is_available():
        return
        
    # Optimize TensorFloat32 usage
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Use cudnn benchmark for optimized performance
    torch.backends.cudnn.benchmark = True

# After the imports section, add this helper function for model downloading
def download_model_to_cache(model_id, cache_dir=None):
    """
    Download a model from HuggingFace to a local cache directory.
    This function can be used to pre-download models for offline use.
    """
    import os
    import requests
    import json
    from tqdm import tqdm
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "transformers")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Directory for this specific model
    model_dir = os.path.join(cache_dir, model_id.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a marker file to indicate this is a manually downloaded model
    with open(os.path.join(model_dir, "offline_download.json"), "w") as f:
        json.dump({"model_id": model_id, "download_date": str(time.time())}, f)
    
    # Get model files list from HuggingFace API
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}/tree/main")
        if r.status_code == 200:
            files = r.json()
            # Download each required file
            for file in tqdm(files, desc=f"Downloading {model_id}"):
                if file["type"] == "file" and (file["path"].endswith(".json") or 
                                               file["path"].endswith(".bin") or 
                                               file["path"].endswith(".model")):
                    file_url = f"https://huggingface.co/{model_id}/resolve/main/{file['path']}"
                    local_path = os.path.join(model_dir, os.path.basename(file["path"]))
                    
                    # Skip if file already exists
                    if os.path.exists(local_path):
                        continue
                    
                    # Download file
                    with requests.get(file_url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
        else:
            logging.error(f"Failed to get file list: {r.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        return False
    
    return model_dir

class LocalModelHandler:
    """Handles loading and inference with local LLM models"""
    
    def __init__(self, model_path="NousResearch/DeepHermes-3-Mistral-24B-Preview"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.gpu_id = 0  # Default to first GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.offline_mode = False
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "transformers")
        self.compiled_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "compiled_models")
        self.use_compiled_cache = True
        
        # Create compiled cache dir if it doesn't exist
        os.makedirs(self.compiled_cache_dir, exist_ok=True)
    
    def set_gpu_device(self, gpu_id):
        """Set the specific GPU to use by ID"""
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.gpu_id = gpu_id
            torch.cuda.set_device(gpu_id)
            self.device = f"cuda:{gpu_id}"
            gpu_name = torch.cuda.get_device_name(gpu_id)
            logging.info(f"GPU set to: {gpu_id} - {gpu_name}")
            return True, f"GPU set to: {gpu_name}"
        return False, "Invalid GPU ID or CUDA not available"
        
    def set_offline_mode(self, enabled):
        """Enable or disable offline mode"""
        self.offline_mode = enabled
        logging.info(f"Offline mode: {enabled}")
        
    def set_cache_dir(self, directory):
        """Set the cache directory for models"""
        if directory and os.path.isdir(directory):
            self.cache_dir = directory
            logging.info(f"Cache directory set to: {directory}")
            return True
        return False
        
    def check_model_path(self):
        """Verify if the model path is valid"""
        try:
            # Check if it's a local path
            if os.path.exists(self.model_path):
                # Check if the directory contains the expected files
                expected_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
                # Note: actual files may vary depending on model structure
                files_present = [f for f in expected_files if os.path.exists(os.path.join(self.model_path, f))]
                
                if files_present:
                    return True, f"Found {len(files_present)} expected model files"
                return False, "Path exists but doesn't contain model files"
            
            # If offline mode, check if model exists in cache
            if self.offline_mode:
                # Convert model ID to cache path format (e.g., "org/model" -> "org--model")
                cached_path = os.path.join(self.cache_dir, self.model_path.replace("/", "--"))
                
                if os.path.exists(cached_path):
                    return True, f"Model found in cache: {cached_path}"
                return False, f"Model not found in cache directory: {cached_path}"
            
            # If not a local path, assume it's a HuggingFace model ID
            # We can't check fully without loading, but we can do basic validation
            if "/" not in self.model_path:
                return False, "Model path should be either a local path or HuggingFace model ID (org/model)"
                
            return True, "Using HuggingFace model ID"
        except Exception as e:
            return False, f"Error checking model path: {str(e)}"
    
    def get_system_info(self):
        """Get system information for diagnostics"""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
        }
        
        if torch.cuda.is_available():
            info["current_gpu_id"] = self.gpu_id
            info["current_gpu"] = torch.cuda.get_device_name(self.gpu_id)
            info["cuda_version"] = torch.version.cuda
            
            # Get all available GPUs
            gpus = []
            for i in range(torch.cuda.device_count()):
                gpus.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            info["available_gpus"] = gpus
            
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(self.gpu_id)
                info["gpu_memory"] = f"{free_mem / (1024**3):.2f}GB free / {total_mem / (1024**3):.2f}GB total"
            except:
                # Some torch versions don't have mem_get_info
                info["gpu_memory"] = "Unknown"
        
        return info
        
    def get_compiled_model_path(self):
        """Get path for the compiled model cache"""
        # Create a unique identifier based on model path and gpu
        model_id = self.model_path.replace("/", "--")
        gpu_name = torch.cuda.get_device_name(self.gpu_id).replace(" ", "_")
        # Include quantization info and torch version for compatibility
        compiled_name = f"{model_id}__{gpu_name}__4bit__{torch.__version__}.pt"
        return os.path.join(self.compiled_cache_dir, compiled_name)
        
    def initialize(self):
        """Load the model and tokenizer with enhanced error handling"""
        logging.info(f"Loading model from {self.model_path} on {self.device}")
        
        # Check model path before loading
        path_ok, path_message = self.check_model_path()
        if not path_ok:
            logging.error(f"Invalid model path: {path_message}")
            return False, f"Invalid model path: {path_message}"
            
        # Get system info for diagnostics
        sys_info = self.get_system_info()
        logging.info(f"System info: {sys_info}")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logging.error("CUDA not available. Cannot use GPU.")
            return False, "CUDA not available. Cannot use GPU."
        
        # Set the specific GPU device
        torch.cuda.set_device(self.gpu_id)
        
        # Apply GPU optimizations
        optimize_gpu_settings()
        
        # Display memory info for the selected GPU
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(self.gpu_id)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            logging.info(f"GPU {self.gpu_id} memory: {free_gb:.2f}GB free / {total_gb:.2f}GB total")
                
            # Warning if free memory is too low
            min_memory_gb = 8  # RTX 3070 Ti should have at least 8GB
            if free_gb < min_memory_gb:
                logging.warning(f"Low GPU memory ({free_gb:.2f}GB free). Model may not fit. Consider 4-bit quantization.")
        except:
            logging.info("Could not get detailed GPU memory information")
        
        try:
            # Implement more specific error handling during model loading
            try:
                logging.info("Loading tokenizer...")
                
                # Determine actual model path based on input path and offline mode
                if self.offline_mode:
                    # Convert model ID to cache path format if it's not a local path
                    if not os.path.exists(self.model_path):
                        model_path = os.path.join(self.cache_dir, self.model_path.replace("/", "--"))
                        if not os.path.exists(model_path):
                            return False, f"Model not found in cache directory: {model_path}. Pre-download the model first."
                    else:
                        model_path = self.model_path
                    logging.info(f"Using local model path: {model_path}")
                else:
                    # Online mode - use the original model path
                    model_path = self.model_path
                    logging.info(f"Using HuggingFace model path: {model_path}")
                
                # Check if we have a cached compiled model
                compiled_path = self.get_compiled_model_path()
                has_compiled_model = os.path.exists(compiled_path) and self.use_compiled_cache
                
                # Optimization - first try to load from compiled cache for faster startup
                if has_compiled_model:
                    logging.info(f"Found pre-compiled model at {compiled_path}")
                    try:
                        start_time = time.time()
                        
                        # Load tokenizer first
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path,
                            local_files_only=self.offline_mode
                        )
                        
                        # Fix pad token if needed
                        if self.tokenizer.pad_token is None:
                            logging.info("Setting pad_token to eos_token")
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                        # Load compiled model
                        logging.info("Loading pre-compiled model from cache...")
                        self.model = torch.load(compiled_path)
                        load_time = time.time() - start_time
                        logging.info(f"Compiled model loaded in {load_time:.2f}s")
                        self.initialized = True
                        return True, f"Compiled model loaded in {load_time:.2f}s"
                    except Exception as e:
                        logging.warning(f"Failed to load compiled model: {e}. Falling back to standard loading.")
                        # Continue with standard loading path
                
                # Create 4-bit quantization config with optimized settings
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage=torch.uint8
                )
                
                # Standard loading path for both offline and online modes
                start_time = time.time()
                
                # Check if model has safetensors format
                has_safetensors = False
                if self.offline_mode:
                    # Check if model.safetensors exists in the model directory
                    safetensors_path = os.path.join(model_path, "model.safetensors")
                    if os.path.exists(safetensors_path):
                        has_safetensors = True
                        logging.info("Found safetensors format")
                
                # Common loading parameters
                load_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": {"": self.gpu_id},
                    "quantization_config": quantization_config,
                    "trust_remote_code": True,
                    # Set use_safetensors to False if we know safetensors doesn't exist
                    "use_safetensors": has_safetensors if self.offline_mode else None,
                }
                
                # Add offline/online specific parameters
                if self.offline_mode:
                    load_kwargs["local_files_only"] = True
                else:
                    load_kwargs["cache_dir"] = self.cache_dir
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=self.offline_mode
                )
                
                # Fix pad token if needed
                if self.tokenizer.pad_token is None:
                    logging.info("Setting pad_token to eos_token")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with appropriate parameters
                logging.info(f"Loading model with 4-bit quantization ({model_path})...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **load_kwargs
                    )
                except Exception as first_load_error:
                    # If the first attempt fails and we didn't explicitly set use_safetensors=False,
                    # try again with use_safetensors=False
                    if "model.safetensors" in str(first_load_error) or "safetensor" in str(first_load_error).lower():
                        logging.warning("Failed to load with safetensors, trying with .bin format...")
                        load_kwargs["use_safetensors"] = False
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **load_kwargs
                        )
                    else:
                        # Re-raise the original error if it wasn't about safetensors
                        raise
                
                load_time = time.time() - start_time
                logging.info(f"Model loaded in {load_time:.2f}s")
                
                # If we loaded successfully and didn't use a compiled model, save the model to the cache
                if self.use_compiled_cache and not has_compiled_model:
                    try:
                        logging.info(f"Saving compiled model to {compiled_path} for faster future loading")
                        torch.save(self.model, compiled_path)
                        logging.info("Compiled model saved successfully")
                    except Exception as e:
                        logging.error(f"Failed to save compiled model: {e}")
                
            except ImportError as e:
                if "bitsandbytes" in str(e).lower():
                    logging.error("Missing bitsandbytes library required for 4-bit quantization")
                    return False, "Missing bitsandbytes library. Install with: pip install bitsandbytes"
                raise
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.error("GPU out of memory during model loading")
                    return False, "GPU out of memory. Try closing other applications or use a smaller model."
                raise
            
            except ValueError as e:
                if "huggingface" in str(e).lower() and "token" in str(e).lower():
                    logging.error("HuggingFace authentication error")
                    return False, "HuggingFace authentication required. Enable offline mode and use a pre-downloaded model."
                raise
                
            except Exception as e:
                # Re-raise for general handler
                raise
            
            self.initialized = True
            logging.info("Model loaded successfully")
            return True, "Model loaded successfully"
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to initialize model: {error_msg}")
            
            # Provide more helpful error messages based on common issues
            if "No such file or directory" in error_msg:
                return False, f"Model not found: {self.model_path}"
            elif "CUDA" in error_msg and "out of memory" in error_msg:
                return False, "GPU memory insufficient. Try a smaller model or 4-bit quantization."
            elif "CPU" in error_msg and "too many dimensions" in error_msg:
                return False, "Model too large for CPU. Use GPU or a smaller model."
            elif "huggingface.co" in error_msg and ("token" in error_msg or "permission" in error_msg):
                return False, "HuggingFace authentication error. Use offline mode with a pre-downloaded model."
            else:
                return False, f"Model initialization error: {error_msg}"
    
    def generate_text(self, prompt, settings=None):
        """Generate text based on prompt using the local model"""
        if not self.initialized:
            success, message = self.initialize()
            if not success:
                return f"Error: Model initialization failed - {message}"
        
        try:
            # Apply settings
            temperature = settings.get("temperature", 0.7) if settings else 0.7
            max_length = settings.get("max_length", 2048) if settings else 2048
            top_p = settings.get("top_p", 0.9) if settings else 0.9
            top_k = settings.get("top_k", 50) if settings else 50
            
            # Prepare inputs with proper attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # Hard limit to prevent context overflows
            ).to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Explicitly pass attention mask
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id  # Explicitly set pad token ID
            )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):]
                
            return response.strip()
            
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return f"Error during generation: {str(e)}"
    
    def unload(self):
        """Unload the model to free memory"""
        if self.initialized:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.initialized = False

    def update_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            try:
                # Get memory for the specific GPU
                free_mem, total_mem = torch.cuda.mem_get_info(self.gpu_id)
                used_mem = total_mem - free_mem
                return {
                    "free": free_mem / (1024**3),
                    "total": total_mem / (1024**3),
                    "used": used_mem / (1024**3)
                }
            except:
                pass
        return None

# ------------------------------------------------------------------------------
# Global Variables and Initial Data Loading
local_model = None
instructions_manager = None
try:
    # Initialize local model handler - actual loading happens on first use
    local_model = LocalModelHandler()
    logging.info("Local model handler initialized")
    
    # Initialize instructions manager
    instructions_manager = None
    try:
        # Create the parent directory if it doesn't exist
        instructions_path = os.path.join(os.path.expanduser("~"), "TrueHalt", "system_instructions.json")
        os.makedirs(os.path.dirname(instructions_path), exist_ok=True)
        
        # Initialize instructions manager with correct path
        instructions_manager = InstructionsManager(instructions_path)
        logging.info(f"Instructions manager initialized with path: {instructions_path}")
    except Exception as e:
        logging.error(f"Failed to initialize instructions manager: {e}")
        # Create a basic instructions manager with default instructions
        try:
            instructions_manager = InstructionsManager(None)
            instructions_manager.set_system_instruction("General Assistant", 
                "You are a helpful, respectful assistant. Provide accurate, detailed responses.")
            logging.info("Created fallback instructions manager")
        except:
            logging.error("Failed to create fallback instructions manager")
except Exception as e:
    logging.error(f"Failed to initialize components: {e}")

# Add global theme settings
current_theme = "dark"  # Default theme
themes = {
    "dark": {
        "background": "#1D1D1F",
        "text": "#F5F5F7",
        "input_bg": "#2C2C2E",
        "output_bg": "#2C2C2E",
        "accent": "#147EFB",
        "secondary": "#8E8E93"
    },
    "light": {
        "background": "#F5F5F7",
        "text": "#1D1D1F",
        "input_bg": "#FFFFFF",
        "output_bg": "#FFFFFF",
        "accent": "#007AFF",
        "secondary": "#8E8E93"
    }
}

# Add current fonts dictionary
current_fonts = {
    "input": QFont("Consolas", 11),
    "output": QFont("Consolas", 11),
    "ui": QFont("Segoe UI", 10)
}

# Dictionary to store chat histories
chat_histories = {}

model_settings = {
    "model": "NousResearch/DeepHermes-3-Mistral-24B-Preview",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_length": 2048
}
stop_event = Event()

# Add a Toast notification class for UI feedback
class Toast(QObject):
    """Simple toast notification class for displaying temporary messages"""
    
    @staticmethod
    def show(parent, message, duration=2000):
        """
        Show a toast message that automatically disappears after duration
        
        Args:
            parent: Parent widget
            message: Message to display
            duration: Duration in milliseconds
        """
        toast = QLabel(message, parent)
        toast.setAlignment(Qt.AlignCenter)
        toast.setStyleSheet(f"""
            background-color: {themes[current_theme].get('secondary', '#8E8E93')};
            color: {themes[current_theme].get('text', '#FFFFFF')};
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 14px;
        """)
        toast.adjustSize()
        
        # Position at the bottom of the parent
        parent_rect = parent.geometry()
        toast_x = (parent_rect.width() - toast.width()) // 2
        toast_y = parent_rect.height() - toast.height() - 40
        toast.move(toast_x, toast_y)
        
        # Show toast
        toast.show()
        
        # Timer to hide and delete toast
        def hide_toast():
            toast.hide()
            toast.deleteLater()
            
        QTimer.singleShot(duration, hide_toast)

# Add a worker class for thread-safe model inference
class LocalModelWorker(QObject):
    generation_complete = pyqtSignal(str)
    generation_error = pyqtSignal(str)
    
    def __init__(self, prompt, model_settings):
        super().__init__()
        self.prompt = prompt
        self.model_settings = model_settings
    
    @pyqtSlot()
    def generate(self):
        try:
            # Check if model is available
            if not local_model:
                self.generation_error.emit("Local model handler not initialized.")
                return
            
            # Generate text
            response = local_model.generate_text(self.prompt, self.model_settings)
            
            # Check if we have a valid response
            if response and not response.startswith("Error"):
                output_text = response
            else:
                output_text = response if response else "No response generated."
            
            # Emit the result signal
            self.generation_complete.emit(output_text)
            
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            self.generation_error.emit(str(e))

# Add a worker class for multi-agent dialog
class LocalDialogWorker(QObject):
    agent_response = pyqtSignal(int, str)
    dialog_complete = pyqtSignal()
    dialog_error = pyqtSignal(str)
    
    def __init__(self, prompt, agent_roles, num_agents, continuous_mode=False, max_turns=None):
        super().__init__()
        self.prompt = prompt
        self.agent_roles = agent_roles
        self.num_agents = num_agents
        self.conversation_history = []
        self.continuous_mode = continuous_mode
        self.max_turns = max_turns
        self.current_turn = 0
        self.stop_requested = False
        
    def request_stop(self):
        """Request the dialog to stop after current agent completes"""
        self.stop_requested = True
    
    @pyqtSlot()
    def generate_dialog(self):
        """Generate a conversation between multiple agents"""
        try:
            # Check if model is available
            global local_model
            if not local_model:
                self.dialog_error.emit("Local model handler not initialized.")
                return
                
            # Initialize model if needed
            if not local_model.initialized:
                success = local_model.initialize()
                if not success:
                    self.dialog_error.emit("Failed to initialize the model.")
                    return
            
            # Add user query to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": self.prompt
            })
            
            # Run initial agent responses
            agent_idx = 0
            
            # Continue until stopped or max turns reached
            while not self.stop_requested and (self.max_turns is None or self.current_turn < self.max_turns):
                # Get agent role description
                agent_role = self.agent_roles.get(agent_idx, f"Agent {agent_idx+1} analyzing and responding to previous content.")
                
                # Build the prompt including conversation history
                agent_prompt = f"You are Agent {agent_idx+1}. {agent_role}\n\n"
                agent_prompt += "User Query: " + self.prompt + "\n\n"
                
                # Add previous agent responses
                if len(self.conversation_history) > 1:
                    agent_prompt += "Previous responses:\n"
                    # Include more context for continuous mode
                    max_history = 10 if self.continuous_mode else len(self.conversation_history)
                    history_to_include = self.conversation_history[-max_history:] if len(self.conversation_history) > max_history else self.conversation_history
                    
                    for entry in history_to_include:
                        role_name = entry["role"]
                        agent_prompt += f"{role_name}: {entry['content']}\n\n"
                
                # Generate the agent's response
                agent_response = local_model.generate_text(agent_prompt, {
                    "temperature": 0.7,
                    "max_length": 1024,  # Shorter for multi-agent dialogs
                    "top_p": 0.9,
                    "top_k": 40
                })
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": f"Agent {agent_idx+1}",
                    "content": agent_response
                })
                
                # Emit the response signal
                self.agent_response.emit(agent_idx, agent_response)
                
                # If not in continuous mode, break after all agents have responded once
                if not self.continuous_mode:
                    if agent_idx == self.num_agents - 1:
                        break
                
                # Move to next agent in rotation
                agent_idx = (agent_idx + 1) % self.num_agents
                
                # Increment turn counter when we've gone through all agents
                if agent_idx == 0:
                    self.current_turn += 1
                
                # Small delay to allow UI to update
                time.sleep(0.5)
                
            # Signal completion of the multi-agent dialog
            self.dialog_complete.emit()
            
        except Exception as e:
            logging.error(f"Failed to generate multi-agent dialog: {e}")
            self.dialog_error.emit(str(e))

# ------------------------------------------------------------------------------
# Main Application Class
class TrueHaltApp(QMainWindow):
    model_status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrueHalt - Local LLM Interface")
        self.resize(1200, 800)
        
        # Create main UI elements
        self.create_ui()
        
        # Apply theme
        self.apply_theme()
        
        # Initialize chat pages
        self.input_entries = []
        self.output_texts = []
        self.add_chat_page("Chat 1")
        
        # Connect model status signal
        self.model_status_signal.connect(self.update_model_status)
        
    def create_ui(self):
        # Main layout with splitter
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        
        # Create horizontal splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side - chat area
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        
        # Add new tab button
        self.tab_widget.setCornerWidget(self.create_new_tab_button())
        
        self.splitter.addWidget(self.tab_widget)
        
        # Right side - settings
        self.settings_tabs = QTabWidget()
        self.setup_settings_tabs()
        
        self.splitter.addWidget(self.settings_tabs)
        self.splitter.setSizes([700, 300])  # Default split sizes
        
        main_layout.addWidget(self.splitter)
        
        # Status bar elements
        status_bar = QFrame()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_left = QLabel("Ready")
        status_layout.addWidget(self.status_left)
        
        # Progress bar
        self.progress_bar = QFrame()
        progress_layout = QHBoxLayout(self.progress_bar)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_indicator = QLabel()
        self.progress_indicator.setFixedHeight(4)
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 0%;
        """)
        progress_layout.addWidget(self.progress_indicator)
        self.progress_bar.hide()
        
        status_layout.addWidget(self.progress_bar)
        
        # Right status area
        self.status_right = QHBoxLayout()
        status_layout.addLayout(self.status_right)
        
        main_layout.addWidget(status_bar)
    
    def create_new_tab_button(self):
        """Create a + button to add new tabs"""
        btn = QToolButton()
        btn.setText("+")
        btn.setToolTip("Add new chat tab")
        btn.clicked.connect(self.add_new_chat_tab)
        return btn
    
    def add_new_chat_tab(self):
        """Add a new chat tab"""
        tab_count = self.tab_widget.count() + 1
        self.add_chat_page(f"Chat {tab_count}")
    
    def add_chat_page(self, title):
        """Add a new chat page with input and output areas"""
        chat_page = QWidget()
        layout = QVBoxLayout(chat_page)
        
        # Chat output (conversation history)
        output_text = QTextEdit()
        output_text.setReadOnly(True)
        output_text.setFont(current_fonts["output"])
        layout.addWidget(output_text, 7)  # 70% of space
        
        # Input area
        input_entry = QTextEdit()
        input_entry.setFont(current_fonts["input"])
        input_entry.setPlaceholderText("Type your message here...")
        layout.addWidget(input_entry, 2)  # 20% of space
        
        # Button area
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: input_entry.clear())
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        
        submit_btn = QPushButton("Generate Response")
        submit_btn.clicked.connect(lambda: self.generate_response(self.tab_widget.indexOf(chat_page)))
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout, 1)  # 10% of space
        
        # Store references to input and output
        self.input_entries.append(input_entry)
        self.output_texts.append(output_text)
        
        # Add to tab widget
        self.tab_widget.addTab(chat_page, title)
        self.tab_widget.setCurrentWidget(chat_page)
    
    def close_tab(self, index):
        """Close the specified tab"""
        if self.tab_widget.count() > 1:  # Keep at least one tab
            # Remove references to input and output
            del self.input_entries[index]
            del self.output_texts[index]
            
            # Remove tab
            self.tab_widget.removeTab(index)
    
    def setup_settings_tabs(self):
        """Create the settings tabs on the right panel"""
        # Chat Settings Tab
        chat_settings_tab = QWidget()
        chat_settings_layout = QVBoxLayout(chat_settings_tab)
        
        # Multi-response mode
        multi_response_frame = QFrame()
        multi_response_layout = QHBoxLayout(multi_response_frame)
        
        self.multi_response_check = QCheckBox("Multi Response Mode")
        self.multi_response_check.toggled.connect(self.toggle_multi_response_mode)
        
        self.response_count_label = QLabel("Number of responses:")
        self.response_count_label.setVisible(False)
        
        self.response_count_combo = QComboBox()
        for i in range(1, 11):
            self.response_count_combo.addItem(str(i))
        self.response_count_combo.setCurrentIndex(0)
        self.response_count_combo.setVisible(False)
        
        multi_response_layout.addWidget(self.multi_response_check)
        multi_response_layout.addWidget(self.response_count_label)
        multi_response_layout.addWidget(self.response_count_combo)
        multi_response_layout.addStretch()
        
        chat_settings_layout.addWidget(multi_response_frame)
        
        # Model Settings
        settings_frame = QFrame()
        settings_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        settings_layout = QVBoxLayout(settings_frame)
        
        settings_layout.addWidget(QLabel("Model Settings"))
        
        # Temperature
        temp_frame = QFrame()
        temp_layout = QHBoxLayout(temp_frame)
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temperature_entry = QLineEdit(str(model_settings.get("temperature", 0.7)))
        temp_layout.addWidget(self.temperature_entry)
        settings_layout.addWidget(temp_frame)

        # Top P
        top_p_frame = QFrame()
        top_p_layout = QHBoxLayout(top_p_frame)
        top_p_layout.addWidget(QLabel("Top P:"))
        self.top_p_entry = QLineEdit(str(model_settings["top_p"]))
        top_p_layout.addWidget(self.top_p_entry)
        settings_layout.addWidget(top_p_frame)

        # Top K
        top_k_frame = QFrame()
        top_k_layout = QHBoxLayout(top_k_frame)
        top_k_layout.addWidget(QLabel("Top K:"))
        self.top_k_entry = QLineEdit(str(model_settings.get("top_k", 40)))
        top_k_layout.addWidget(self.top_k_entry)
        settings_layout.addWidget(top_k_frame)
        
        # Max Length
        max_length_frame = QFrame()
        max_length_layout = QHBoxLayout(max_length_frame)
        max_length_layout.addWidget(QLabel("Max Length:"))
        self.max_length_entry = QLineEdit(str(model_settings.get("max_length", 2048)))
        max_length_layout.addWidget(self.max_length_entry)
        settings_layout.addWidget(max_length_frame)

        # Add thinking process toggle
        thinking_frame = QFrame()
        thinking_layout = QHBoxLayout(thinking_frame)
        self.show_thinking_checkbox = QCheckBox("Show Step-by-Step Thinking")
        self.show_thinking_checkbox.setToolTip("Enables explicit reasoning steps for problem solving")
        self.show_thinking_checkbox.setChecked(model_settings.get("show_thinking", False))
        thinking_layout.addWidget(self.show_thinking_checkbox)
        settings_layout.addWidget(thinking_frame)
        
        # Model Selection
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Select Model:"))
        
        self.model_selector = QComboBox()
        models = [
            "NousResearch/DeepHermes-3-Mistral-24B-Preview",
            "NousResearch/Hermes-3-Llama-3.1-8B",
            "NousResearch/Hermes-2-Pro-Mistral-7B",
            "cognitivecomputations/Wizard-Vicuna-7B-Uncensored",
             "cognitivecomputationsWizardLM-1.0-Uncensored-Llama2-13b",  # Updated to full model name
            # Add any other local models you want to support here
        ]
        self.model_selector.addItems(models)
        self.model_selector.setCurrentText(model_settings["model"])
        
        model_layout.addWidget(self.model_selector)
        settings_layout.addWidget(model_frame)
        
        # Model Status
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.addWidget(QLabel("Model Status:"))
        
        self.model_status_indicator = QLabel("●")
        self.model_status_indicator.setStyleSheet("color: #8E8E93;")
        
        self.model_status_label = QLabel("Not loaded")
        
        check_model_btn = QPushButton("Check Model")
        check_model_btn.clicked.connect(self.check_model_status)
        
        status_layout.addWidget(self.model_status_indicator)
        status_layout.addWidget(self.model_status_label)
        status_layout.addWidget(check_model_btn)
        settings_layout.addWidget(status_frame)
        
        # Apply button
        self.apply_settings_button = QPushButton("Apply Settings")
        self.apply_settings_button.clicked.connect(self.apply_settings)
        settings_layout.addWidget(self.apply_settings_button)
        
        chat_settings_layout.addWidget(settings_frame)
        chat_settings_layout.addStretch()
        
        # Add to settings tab widget
        self.settings_tabs.addTab(chat_settings_tab, "Chat")
        
        # Instructions Tab
        instructions_tab = QWidget()
        instructions_layout = QVBoxLayout(instructions_tab)
        
        # System Instructions
        system_frame = QFrame()
        system_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        system_layout = QVBoxLayout(system_frame)
        
        system_layout.addWidget(QLabel("System Instructions"))
        
        # Add instruction preset dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Instruction Presets:"))
        
        self.preset_dropdown = QComboBox()
        
        # Populate with presets from the instructions manager
        if instructions_manager:
            presets = instructions_manager.get_system_instruction_names()
            self.preset_dropdown.addItems(presets)
            # Set to current instruction
            current = instructions_manager.instructions.get("current_system_instruction")
            if current in presets:
                self.preset_dropdown.setCurrentText(current)
        
        preset_layout.addWidget(self.preset_dropdown)
        
        self.apply_preset_btn = QPushButton("Apply")
        self.apply_preset_btn.clicked.connect(self.apply_preset)
        preset_layout.addWidget(self.apply_preset_btn)
        
        system_layout.addLayout(preset_layout)
        
        # System instruction text editor
        system_layout.addWidget(QLabel("Edit system instructions:"))
        
        self.system_instructions_text = QTextEdit()
        self.system_instructions_text.setFont(current_fonts["input"])
        self.system_instructions_text.setMinimumHeight(150)
        
        # Load current instruction text
        if instructions_manager:
            self.system_instructions_text.setPlainText(
                instructions_manager.get_current_system_instruction()
            )
        
        system_layout.addWidget(self.system_instructions_text)
        
        instructions_layout.addWidget(system_frame)
        
        # Developer Instructions
        developer_frame = QFrame()
        developer_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        developer_layout = QVBoxLayout(developer_frame)
        
        developer_layout.addWidget(QLabel("Developer Instructions"))
        developer_layout.addWidget(QLabel("Set developer instructions:"))
        
        self.developer_instructions_text = QTextEdit()
        self.developer_instructions_text.setFont(current_fonts["input"])
        self.developer_instructions_text.setMinimumHeight(150)
        developer_layout.addWidget(self.developer_instructions_text)
        
        instructions_layout.addWidget(developer_frame)
        
        # Save and Custom sections
        save_frame = QFrame()
        save_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        save_layout = QVBoxLayout(save_frame)
        
        # Custom preset section
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Save as Custom Preset:"))
        
        self.custom_preset_name = QLineEdit()
        custom_layout.addWidget(self.custom_preset_name)
        
        save_custom_btn = QPushButton("Save Custom")
        save_custom_btn.clicked.connect(self.save_custom_preset)
        custom_layout.addWidget(save_custom_btn)
        
        save_layout.addLayout(custom_layout)
        
        # Save current button
        save_current_btn = QPushButton("Save Current Instruction")
        save_current_btn.clicked.connect(self.save_system_instructions)
        save_layout.addWidget(save_current_btn)
        
        instructions_layout.addWidget(save_frame)
        instructions_layout.addStretch()
        
        self.settings_tabs.addTab(instructions_tab, "Instructions")
        
        # Add other tabs (Appearance, Tools, Agents)
        self.add_appearance_tab()
        self.add_tools_tab()
        self.add_agents_tab()

    def add_tools_tab(self):
        """Add tools tab with file operations"""
        tools_tab = QWidget()
        tools_layout = QVBoxLayout(tools_tab)
        
        # Environment section
        env_frame = QFrame()
        env_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        env_layout = QVBoxLayout(env_frame)
        
        env_layout.addWidget(QLabel("Environment"))
        
        load_env_button = QPushButton("Load .env File")
        load_env_button.clicked.connect(self.load_env_file)
        env_layout.addWidget(load_env_button)
        
        tools_layout.addWidget(env_frame)
        
        # Model Management
        model_frame = QFrame()
        model_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        model_layout = QVBoxLayout(model_frame)
        
        model_layout.addWidget(QLabel("Model Management"))
        
        # GPU selection
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("Select GPU:"))
        self.gpu_selector = QComboBox()
        
        # Get available GPUs
        available_gpus = get_available_gpus()
        if available_gpus:
            for gpu_id, gpu_name in available_gpus:
                self.gpu_selector.addItem(f"GPU {gpu_id}: {gpu_name}", gpu_id)
            
            # Set current GPU to RTX 3070 Ti if available
            for i in range(self.gpu_selector.count()):
                if "3070 Ti" in self.gpu_selector.itemText(i):
                    self.gpu_selector.setCurrentIndex(i)
                    break
        else:
            self.gpu_selector.addItem("No GPUs available", -1)
        
        gpu_layout.addWidget(self.gpu_selector)
        
        gpu_apply_btn = QPushButton("Apply GPU")
        gpu_apply_btn.clicked.connect(self.change_gpu)
        gpu_layout.addWidget(gpu_apply_btn)
        
        model_layout.addLayout(gpu_layout)
        
        # Offline mode toggle
        offline_mode_layout = QHBoxLayout()
        self.offline_mode_check = QCheckBox("Offline Mode (no HuggingFace tokens)")
        self.offline_mode_check.setChecked(local_model.offline_mode if local_model else False)
        self.offline_mode_check.setToolTip("When enabled, only uses local model files with no online access")
        self.offline_mode_check.toggled.connect(self.toggle_offline_mode)
        offline_mode_layout.addWidget(self.offline_mode_check)
        model_layout.addLayout(offline_mode_layout)
        
        # Cache directory
        cache_dir_layout = QHBoxLayout()
        cache_dir_layout.addWidget(QLabel("Cache Directory:"))
        self.cache_dir_entry = QLineEdit(local_model.cache_dir if local_model else "")
        cache_dir_layout.addWidget(self.cache_dir_entry)
        
        cache_dir_browse_btn = QPushButton("Browse...")
        cache_dir_browse_btn.clicked.connect(self.browse_cache_dir)
        cache_dir_layout.addWidget(cache_dir_browse_btn)
        
        model_layout.addLayout(cache_dir_layout)
        
        # Download model button
        download_model_btn = QPushButton("Pre-Download Model")
        download_model_btn.clicked.connect(self.download_model)
        download_model_btn.setToolTip("Download a model for offline use")
        model_layout.addWidget(download_model_btn)
        
        # Add custom model path option
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model Path:"))
        self.model_path_entry = QLineEdit(local_model.model_path if local_model else "")
        model_path_layout.addWidget(self.model_path_entry)
        
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self.browse_model_path)
        model_path_layout.addWidget(model_browse_btn)
        
        model_layout.addLayout(model_path_layout)
        
        # Buttons for model management
        model_buttons_layout = QHBoxLayout()
        
        change_model_button = QPushButton("Change Model")
        change_model_button.clicked.connect(self.change_model_path)
        model_buttons_layout.addWidget(change_model_button)
        
        unload_model_button = QPushButton("Unload Model")
        unload_model_button.clicked.connect(self.unload_model)
        model_buttons_layout.addWidget(unload_model_button)
        
        reload_model_button = QPushButton("Reload Model")
        reload_model_button.clicked.connect(self.reload_model)
        model_buttons_layout.addWidget(reload_model_button)
        
        model_layout.addLayout(model_buttons_layout)
        
        # Memory usage info
        memory_frame = QFrame()
        memory_layout = QHBoxLayout(memory_frame)
        memory_layout.addWidget(QLabel("Memory Usage:"))
        
        self.memory_usage_label = QLabel("Unknown")
        memory_layout.addWidget(self.memory_usage_label)
        
        update_memory_button = QPushButton("Update")
        update_memory_button.clicked.connect(self.update_memory_usage)
        memory_layout.addWidget(update_memory_button)
        
        model_layout.addWidget(memory_frame)
        
        # System info button
        system_info_button = QPushButton("System Information")
        system_info_button.clicked.connect(self.show_system_info)
        model_layout.addWidget(system_info_button)
        
        tools_layout.addWidget(model_frame)
        
        # File Operations section
        file_frame = QFrame()
        file_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        file_layout = QVBoxLayout(file_frame)
        
        file_layout.addWidget(QLabel("File Operations"))
        
        load_file_button = QPushButton("Load Text from File")
        load_file_button.clicked.connect(lambda: self.perform_file_operation("load"))
        file_layout.addWidget(load_file_button)
        
        save_file_button = QPushButton("Save Output to File")
        save_file_button.clicked.connect(lambda: self.perform_file_operation("save"))
        file_layout.addWidget(save_file_button)
        
        save_session_button = QPushButton("Save Current Session")
        save_session_button.clicked.connect(lambda: self.perform_file_operation("save_session"))
        file_layout.addWidget(save_session_button)
        
        load_session_button = QPushButton("Load Saved Session")
        load_session_button.clicked.connect(lambda: self.perform_file_operation("load_session"))
        file_layout.addWidget(load_session_button)
        
        tools_layout.addWidget(file_frame)
        
        # AI Goals section
        goals_frame = QFrame()
        goals_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        goals_layout = QVBoxLayout(goals_frame)
        
        goals_layout.addWidget(QLabel("AI Goals"))
        
        update_goals_button = QPushButton("Update AI Goals")
        update_goals_button.clicked.connect(self.update_goals)
        goals_layout.addWidget(update_goals_button)
        
        view_goals_button = QPushButton("View AI Goals")
        view_goals_button.clicked.connect(self.view_goals)
        goals_layout.addWidget(view_goals_button)
        
        tools_layout.addWidget(goals_frame)
        tools_layout.addStretch()
        
        self.settings_tabs.addTab(tools_tab, "Tools")

    def add_appearance_tab(self):
        """Add appearance settings tab"""
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        
        # Theme selection
        theme_frame = QFrame()
        theme_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        theme_layout = QVBoxLayout(theme_frame)
        
        theme_layout.addWidget(QLabel("Theme Selection"))
        
        theme_selector = QComboBox()
        theme_selector.addItems(["dark", "light"])
        theme_selector.setCurrentText(current_theme)
        theme_selector.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_selector)
        
        appearance_layout.addWidget(theme_frame)
        
        # Font settings
        font_frame = QFrame()
        font_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        font_layout = QVBoxLayout(font_frame)
        
        font_layout.addWidget(QLabel("Font Settings"))
        
        # Input font
        input_font_layout = QHBoxLayout()
        input_font_layout.addWidget(QLabel("Input Font:"))
        input_font_button = QPushButton("Change...")
        input_font_button.clicked.connect(lambda: self.change_font("input"))
        input_font_layout.addWidget(input_font_button)
        font_layout.addLayout(input_font_layout)
        
        # Output font
        output_font_layout = QHBoxLayout()
        output_font_layout.addWidget(QLabel("Output Font:"))
        output_font_button = QPushButton("Change...")
        output_font_button.clicked.connect(lambda: self.change_font("output"))
        output_font_layout.addWidget(output_font_button)
        font_layout.addLayout(output_font_layout)
        
        appearance_layout.addWidget(font_frame)
        
        # UI settings
        ui_frame = QFrame()
        ui_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        ui_layout = QVBoxLayout(ui_frame)
        
        ui_layout.addWidget(QLabel("UI Settings"))
        
        # Accent color
        accent_layout = QHBoxLayout()
        accent_layout.addWidget(QLabel("Accent Color:"))
        accent_button = QPushButton("Change...")
        accent_button.clicked.connect(self.change_accent_color)
        accent_layout.addWidget(accent_button)
        ui_layout.addLayout(accent_layout)
        
        # Layout options
        layout_selector = QComboBox()
        layout_selector.addItems(["Standard", "Compact", "Expanded"])
        layout_selector.setCurrentText("Standard")
        layout_selector.currentTextChanged.connect(self.change_layout)
        
        layout_option = QHBoxLayout()
        layout_option.addWidget(QLabel("Layout Style:"))
        layout_option.addWidget(layout_selector)
        ui_layout.addLayout(layout_option)
        
        appearance_layout.addWidget(ui_frame)
        appearance_layout.addStretch()
        
        self.settings_tabs.addTab(appearance_tab, "Appearance")

    def add_agents_tab(self):
        """Add agents tab with agent settings"""
        agents_tab = QWidget()
        agents_layout = QVBoxLayout(agents_tab)
        
        # Enable agents toggle
        agent_enable_frame = QFrame()
        agent_enable_layout = QHBoxLayout(agent_enable_frame)
        
        self.agent_enabled = False
        self.agent_toggle = QCheckBox("Enable Multi-Agent Mode")
        self.agent_toggle.setChecked(self.agent_enabled)
        self.agent_toggle.toggled.connect(self.toggle_agent_mode)
        
        agent_enable_layout.addWidget(self.agent_toggle)
        agents_layout.addWidget(agent_enable_frame)
        
        # Agent settings
        agent_settings_frame = QFrame()
        agent_settings_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        agent_settings_layout = QVBoxLayout(agent_settings_frame)
        
        agent_settings_layout.addWidget(QLabel("Agent Settings"))
        
        # Number of agents
        agent_count_layout = QHBoxLayout()
        agent_count_layout.addWidget(QLabel("Number of Agents:"))
        
        self.agent_count_spinner = QSpinBox()
        self.agent_count_spinner.setRange(2, 5)
        self.agent_count_spinner.setValue(2)
        self.agent_count_spinner.valueChanged.connect(self.update_agent_roles)
        
        agent_count_layout.addWidget(self.agent_count_spinner)
        agent_settings_layout.addLayout(agent_count_layout)
        
        # Continuous mode
        continuous_layout = QHBoxLayout()
        self.continuous_mode = QCheckBox("Continuous Mode")
        self.continuous_mode.setToolTip("Agents will continue discussing until stopped")
        continuous_layout.addWidget(self.continuous_mode)
        
        # Maximum turns
        continuous_layout.addWidget(QLabel("Maximum Turns:"))
        self.max_turns = QSpinBox()
        self.max_turns.setRange(1, 20)
        self.max_turns.setValue(5)
        continuous_layout.addWidget(self.max_turns)
        
        agent_settings_layout.addLayout(continuous_layout)
        
        # Agent roles
        agent_settings_layout.addWidget(QLabel("Agent Roles:"))
        
        self.agent_roles_layout = QVBoxLayout()
        agent_settings_layout.addLayout(self.agent_roles_layout)
        
        # Initialize with default roles for 2 agents
        self.agent_role_entries = []
        self.update_agent_roles(2)
        
        agents_layout.addWidget(agent_settings_frame)
        
        # Agent presets
        agent_presets_frame = QFrame()
        agent_presets_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        agent_presets_layout = QVBoxLayout(agent_presets_frame)
        
        agent_presets_layout.addWidget(QLabel("Agent Presets"))
        
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Select Preset:"))
        
        self.agent_preset_combo = QComboBox()
        self.agent_preset_combo.addItems([
            "Debate (Pro & Con)",
            "Code Review (Developer & Reviewer)",
            "Creative (Writer & Editor)",
            "Problem Solving (Analyst & Critic)",
            "Custom"
        ])
        preset_layout.addWidget(self.agent_preset_combo)
        
        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self.load_agent_preset)
        preset_layout.addWidget(load_preset_btn)
        
        agent_presets_layout.addLayout(preset_layout)
        agents_layout.addWidget(agent_presets_frame)
        
        # Save custom preset
        save_preset_layout = QHBoxLayout()
        save_preset_layout.addWidget(QLabel("Preset Name:"))
        
        self.preset_name_entry = QLineEdit()
        save_preset_layout.addWidget(self.preset_name_entry)
        
        save_preset_btn = QPushButton("Save Custom Preset")
        save_preset_btn.clicked.connect(self.save_agent_preset)
        save_preset_layout.addWidget(save_preset_btn)
        
        agents_layout.addLayout(save_preset_layout)
        agents_layout.addStretch()
        
        self.settings_tabs.addTab(agents_tab, "Agents")

    def check_model_status(self):
        """Check if the model is loaded and ready"""
        global local_model
        
        if not local_model:
            self.model_status_indicator.setStyleSheet("color: #FF3B30;")  # Red
            self.model_status_label.setText("Error: Model handler not initialized")
            return
        
        # If model not initialized, try to initialize it
        if not local_model.initialized:
            # Check if we're looking for DeepHermes and in offline mode
            using_deephermes = "DeepHermes" in local_model.model_path
            
            # If DeepHermes model, provide special handling
            if using_deephermes:
                # Make sure offline mode is on for DeepHermes
                if not local_model.offline_mode:
                    reply = QMessageBox.question(self, "DeepHermes Model Detected", 
                        f"The DeepHermes model requires offline mode. Would you like to enable it now?",
                        QMessageBox.Yes | QMessageBox.No)
                    
                    if reply == QMessageBox.Yes:
                        self.offline_mode_check.setChecked(True)
                        local_model.set_offline_mode(True)
                        Toast.show(self, "Offline mode enabled for DeepHermes model", 2000)
                
                # Check cache path
                cache_path = os.path.join(local_model.cache_dir, local_model.model_path.replace("/", "--"))
                if not os.path.exists(cache_path):
                    reply = QMessageBox.question(self, "DeepHermes Model Setup Required", 
                        f"The DeepHermes model needs to be downloaded first. Would you like to run the setup script?",
                        QMessageBox.Yes | QMessageBox.No)
                    
                    if reply == QMessageBox.Yes:
                        self.run_deephermes_setup()
                        return
                    else:
                        QMessageBox.warning(self, "Model Not Available", 
                            f"The model '{local_model.model_path}' is not available locally.\n\n"
                            f"Please either:\n"
                            f"1. Run the DeepHermes setup script, or\n"
                            f"2. Set a different model path")
                        return

            self.model_status_indicator.setStyleSheet("color: #FF9500;")  # Yellow/Orange
            self.model_status_label.setText("Loading model...")
            
            # Use a thread to avoid UI freezing
            def load_model_thread():
                success, message = local_model.initialize()
                if success:
                    self.model_status_signal.emit("loaded")
                else:
                    self.model_status_signal.emit(f"error:{message}")
            
            # Create thread and start it
            thread = Thread(target=load_model_thread)
            thread.daemon = True
            thread.start()
        else:
            # Model is already initialized
            self.model_status_indicator.setStyleSheet("color: #34C759;")  # Green
            self.model_status_label.setText("Model loaded and ready")
            self.update_memory_usage()

    def run_deephermes_setup(self):
        """Run the DeepHermes setup script"""
        setup_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "deephermes_setup.py")
        
        # Check if the script exists
        if not os.path.exists(setup_script):
            setup_script = os.path.join("z:", "TrueHalt", "deephermes_setup.py")
            if not os.path.exists(setup_script):
                QMessageBox.critical(self, "Setup Error", 
                    "Could not find the DeepHermes setup script. Please run it manually:\n\n"
                    "python z:\\TrueHalt\\deephermes_setup.py")
                return
        
        # Run the script in a new console window
        if os.name == 'nt':  # Windows
            os.system(f'start cmd /k python "{setup_script}"')
        else:  # Linux/Mac
            os.system(f'gnome-terminal -- python "{setup_script}"')
        
        QMessageBox.information(self, "Setup Started", 
            "The DeepHermes setup script has been started in a new window.\n\n"
            "After the download completes, return to this application and:\n"
            "1. Make sure 'Offline Mode' is checked\n"
            "2. Click 'Check Model' to load the model")

    def update_model_status(self, status):
        """Update the UI with the model status"""
        if status == "loaded":
            self.model_status_indicator.setStyleSheet("color: #34C759;")  # Green
            self.model_status_label.setText("Model loaded and ready")
            self.update_memory_usage()
        elif status.startswith("error:"):
            self.model_status_indicator.setStyleSheet("color: #FF3B30;")  # Red
            error_message = status[6:]  # Remove "error:" prefix
            self.model_status_label.setText(f"Error: {error_message}")
            # Show error dialog with more details
            QMessageBox.critical(self, "Model Load Error", 
                                f"Failed to load model:\n\n{error_message}\n\n"
                                f"System info:\n{local_model.get_system_info() if local_model else 'Not available'}")
        else:
            self.model_status_indicator.setStyleSheet("color: #8E8E93;")  # Gray
            self.model_status_label.setText("Unknown status")

    def unload_model(self):
        """Unload the model to free memory"""
        global local_model
        
        if local_model and local_model.initialized:
            local_model.unload()
            self.model_status_indicator.setStyleSheet("color: #8E8E93;")  # Gray
            self.model_status_label.setText("Model unloaded")
            self.memory_usage_label.setText("0 MB")
            Toast.show(self, "Model unloaded from memory", 1500)
        else:
            Toast.show(self, "Model not loaded", 1500)

    def reload_model(self):
        """Reload the model"""
        global local_model
        
        if local_model:
            # Unload first if needed
            if local_model.initialized:
                local_model.unload()
                
            # Check model status (will trigger loading)
            self.check_model_status()
        else:
            Toast.show(self, "Model handler not initialized", 1500)

    def update_memory_usage(self):
        """Update the memory usage display"""
        if torch.cuda.is_available():
            # Get GPU memory usage for specific GPU
            try:
                gpu_id = local_model.gpu_id if local_model else 0
                free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
                used_mem = total_mem - free_mem
                
                free_gb = free_mem / (1024**3)
                used_gb = used_mem / (1024**3)
                total_gb = total_mem / (1024**3)
                
                self.memory_usage_label.setText(
                    f"GPU {gpu_id}: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({(used_mem/total_mem*100):.1f}%)"
                )
                
                # Set color based on memory usage
                if used_mem / total_mem > 0.9:  # Over 90% usage
                    self.memory_usage_label.setStyleSheet("color: #FF3B30;")  # Red
                elif used_mem / total_mem > 0.7:  # Over 70% usage
                    self.memory_usage_label.setStyleSheet("color: #FF9500;")  # Orange
                else:
                    self.memory_usage_label.setStyleSheet("")  # Default color
                    
            except Exception as e:
                logging.error(f"Error updating memory usage: {e}")
                self.memory_usage_label.setText("GPU memory info unavailable")
        else:
            # No GPU usage tracking for CPU mode
            self.memory_usage_label.setText("CPU mode (memory usage unknown)")

    def animate_progress(self):
        """Animate the progress bar during model inference"""
        width = 0
        try:
            while not stop_event.is_set() and width < 95:
                time.sleep(0.1)
                width += 1
                # Use QMetaObject.invokeMethod for thread-safe UI updates
                QMetaObject.invokeMethod(
                    self.progress_indicator, 
                    "setStyleSheet", 
                    Qt.QueuedConnection,
                    Q_ARG(str, f"""
                        background-color: {themes[current_theme].get("accent", "#007AFF")};
                        border-radius: 4px;
                        width: {width}%;
                    """)
                )
        except Exception as e:
            logging.error(f"Progress animation error: {e}")
    
    def generate_response(self, page_index):
        """Generate a response based on the input text using the local model"""
        # Check if agent mode is enabled
        if self.agent_enabled:
            self.run_agent(self.input_entries[page_index].toPlainText(), page_index)
            return
                
        # Regular generation process if not in agent mode
        input_text = self.input_entries[page_index].toPlainText()
        if not input_text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to generate a response.")
            return
        
        # Get system instructions if they exist
        system_instructions = ""
        if instructions_manager:
            system_instructions = instructions_manager.get_current_system_instruction()
        elif hasattr(self, 'system_instructions_text'):
            system_instructions = self.system_instructions_text.toPlainText().strip()
        
        # Initialize chat history for this page if it doesn't exist
        if page_index not in chat_histories:
            chat_histories[page_index] = []
        
        # Add the new user message to history
        chat_histories[page_index].append({"role": "user", "content": input_text})
        
        # Prepare prompt with conversation history
        conversation_history = ""
        if len(chat_histories[page_index]) > 1:  # If there's previous conversation
            for msg in chat_histories[page_index][:-1]:  # All messages except current one
                prefix = "User: " if msg["role"] == "user" else "Assistant: "
                conversation_history += f"{prefix}{msg['content']}\n\n"
        
        # Show conversation history in output box
        conversation_display = ""
        for msg in chat_histories[page_index]:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            conversation_display += f"{prefix}{msg['content']}\n\n"
                
        self.output_texts[page_index].setPlainText(conversation_display + "Assistant: Generating...")
        
        # Construct the full prompt with system instructions and conversation history
        prompt = ""
        if system_instructions:
            prompt = f"{system_instructions}\n\n"
        
        if conversation_history:
            prompt += f"Previous conversation:\n{conversation_history}\n"
        
        prompt += f"User: {input_text}\nAssistant:"
        
        # Show progress bar
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 0%;
        """)
        self.progress_bar.show()
        
        # Update status
        self.status_left.setText("Generating response...")
        
        # Start progress animation
        global stop_event
        stop_event.clear()
        progress_thread = Thread(target=self.animate_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Create and run worker in a thread for standard response generation
        self.thread = QThread()
        self.worker = LocalModelWorker(prompt, model_settings)
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.generate)
        self.worker.generation_complete.connect(lambda text: self.handle_standard_response(text, page_index))
        self.worker.generation_error.connect(self.handle_generation_error)
        self.worker.generation_complete.connect(self.thread.quit)
        self.worker.generation_error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Store current page index for the handler
        self.current_page_index = page_index
        
        # Start thread
        self.thread.start()
    
    def handle_standard_response(self, response_text, page_index):
        """Handle the response from the standard generation process"""
        try:
            # Stop the progress animation
            global stop_event
            stop_event.set()
            
            # Hide progress bar with a slight delay
            QTimer.singleShot(500, self.progress_bar.hide)
            
            # Update the progress indicator to 100%
            self.progress_indicator.setStyleSheet(f"""
                background-color: {themes[current_theme].get("accent", "#007AFF")};
                border-radius: 4px;
                width: 100%;
            """)
            
            # Update status
            self.status_left.setText("Response generated")
            
            # Add the assistant response to chat history
            chat_histories[page_index].append({"role": "assistant", "content": response_text})
            
            # Update output display with full conversation
            conversation_display = ""
            for msg in chat_histories[page_index]:
                prefix = "User: " if msg["role"] == "user" else "Assistant: "
                conversation_display += f"{prefix}{msg['content']}\n\n"
                
            self.output_texts[page_index].setPlainText(conversation_display)
            
            # Scroll to the bottom
            cursor = self.output_texts[page_index].textCursor()
            cursor.movePosition(QTextCursor.End)
            self.output_texts[page_index].setTextCursor(cursor)
            
            # Clear input field for next query
            self.input_entries[page_index].clear()
            
        except Exception as e:
            logging.error(f"Error handling response: {e}")
            self.handle_generation_error(str(e))
    
    def handle_generation_error(self, error_message):
        """Handle errors during generation"""
        # Stop the progress animation
        global stop_event
        stop_event.set()
        
        # Hide progress bar
        self.progress_bar.hide()
        
        # Update status
        self.status_left.setText("Error generating response")
        
        # Show error message in output
        page_index = self.current_page_index
        current_output = self.output_texts[page_index].toPlainText()
        
        # Remove "Generating..." message if present
        if current_output.endswith("Assistant: Generating..."):
            current_output = current_output[:-21]
        
        error_msg = f"Error: {error_message}"
        self.output_texts[page_index].setPlainText(current_output + "Assistant: " + error_msg)
        
        # Log the error
        logging.error(f"Generation error: {error_message}")
        
        # Show toast notification
        Toast.show(self, f"Error: {error_message}", 3000)
    
    def run_agent(self, prompt, page_index):
        """Start a multi-agent dialog based on the input text"""
        if not prompt:
            QMessageBox.warning(self, "Input Error", "Please enter some text to generate a response.")
            return
        
        # Get agent roles
        agent_roles = {}
        for i, entry in enumerate(self.agent_role_entries):
            agent_roles[i] = entry.text()
        
        # Get number of agents
        num_agents = self.agent_count_spinner.value()
        
        # Get continuous mode settings
        continuous_mode = self.continuous_mode.isChecked()
        max_turns = self.max_turns.value() if continuous_mode else None
        
        # Show progress bar
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 0%;
        """)
        self.progress_bar.show()
        
        # Update status
        self.status_left.setText("Generating multi-agent dialog...")
        
        # Start progress animation
        global stop_event
        stop_event.clear()
        progress_thread = Thread(target=self.animate_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Clear previous output
        self.output_texts[page_index].clear()
        
        # Create and run worker in a thread for multi-agent dialog
        self.dialog_thread = QThread()
        self.dialog_worker = LocalDialogWorker(prompt, agent_roles, num_agents, continuous_mode, max_turns)
        self.dialog_worker.moveToThread(self.dialog_thread)
        
        # Connect signals
        self.dialog_thread.started.connect(self.dialog_worker.generate_dialog)
        self.dialog_worker.agent_response.connect(lambda idx, text: self.handle_agent_response(idx, text, page_index))
        self.dialog_worker.dialog_complete.connect(self.handle_dialog_complete)
        self.dialog_worker.dialog_error.connect(self.handle_generation_error)
        self.dialog_worker.dialog_complete.connect(self.dialog_thread.quit)
        self.dialog_worker.dialog_error.connect(self.dialog_thread.quit)
        self.dialog_thread.finished.connect(self.dialog_thread.deleteLater)
        
        # Store current page index for the handler
        self.current_page_index = page_index
        
        # Add a stop button
        self.stop_dialog_btn = QPushButton("Stop Dialog")
        self.stop_dialog_btn.clicked.connect(lambda: self.dialog_worker.request_stop())
        self.status_right.addWidget(self.stop_dialog_btn)
        
        # Start thread
        self.dialog_thread.start()
    
    def handle_agent_response(self, agent_idx, response_text, page_index):
        """Handle individual agent responses in multi-agent dialog"""
        try:
            current_output = self.output_texts[page_index].toPlainText()
            
            # Add the agent response with formatting
            formatted_output = f"{current_output}\n\nAgent {agent_idx+1}:\n{response_text}\n"
            
            self.output_texts[page_index].setPlainText(formatted_output)
            
            # Scroll to the bottom
            cursor = self.output_texts[page_index].textCursor()
            cursor.movePosition(QTextCursor.End)
            self.output_texts[page_index].setTextCursor(cursor)
            
        except Exception as e:
            logging.error(f"Error handling agent response: {e}")
    
    def handle_dialog_complete(self):
        """Handle completion of a multi-agent dialog"""
        # Stop the progress animation
        global stop_event
        stop_event.set()
        
        # Hide progress bar with a slight delay
        QTimer.singleShot(500, self.progress_bar.hide)
        
        # Update the progress indicator to 100%
        self.progress_indicator.setStyleSheet(f"""
            background-color: {themes[current_theme].get("accent", "#007AFF")};
            border-radius: 4px;
            width: 100%;
        """)
        
        # Update status
        self.status_left.setText("Multi-agent dialog complete")
        
        # Remove stop button
        if hasattr(self, 'stop_dialog_btn'):
            self.stop_dialog_btn.setParent(None)
            delattr(self, 'stop_dialog_btn')
        
        # Clear input field for next query
        page_index = self.current_page_index
        self.input_entries[page_index].clear()
    
    def perform_file_operation(self, operation):
        """Handle file operations (load/save text or sessions)"""
        try:
            global model_settings
            current_tab = self.tab_widget.currentIndex()
            if operation == "load":
                file_path, _ = QFileDialog.getOpenFileName(self, "Load Text", "", "Text Files (*.txt);;All Files (*)")
                if file_path:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        self.input_entries[current_tab].setPlainText(text)
                    Toast.show(self, f"File loaded: {os.path.basename(file_path)}", 2000)
                    
            elif operation == "save":
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Output", "", "Text Files (*.txt);;All Files (*)")
                if file_path:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(self.output_texts[current_tab].toPlainText())
                    Toast.show(self, f"Output saved: {os.path.basename(file_path)}", 2000)
                    
            elif operation == "save_session":
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON Files (*.json);;All Files (*)")
                if file_path:
                    session_data = {
                        "chat_history": chat_histories.get(current_tab, []),
                        "model_settings": model_settings,
                        "system_instructions": self.system_instructions_text.toPlainText()
                    }
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(session_data, file, indent=2)
                    Toast.show(self, f"Session saved: {os.path.basename(file_path)}", 2000)
                    
            elif operation == "load_session":
                file_path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON Files (*.json);;All Files (*)")
                if file_path:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        session_data = json.load(file)
                        
                    # Apply loaded settings
                    if "chat_history" in session_data:
                        chat_histories[current_tab] = session_data["chat_history"]
                        
                        # Rebuild conversation display
                        conversation_display = ""
                        for msg in chat_histories[current_tab]:
                            prefix = "User: " if msg["role"] == "user" else "Assistant: "
                            conversation_display += f"{prefix}{msg['content']}\n\n"
                        self.output_texts[current_tab].setPlainText(conversation_display)
                        
                    if "model_settings" in session_data:
                        model_settings.update(session_data["model_settings"])
                        
                        # Update UI elements
                        self.temperature_entry.setText(str(model_settings.get("temperature", 0.7)))
                        self.top_p_entry.setText(str(model_settings.get("top_p", 0.9)))
                        self.top_k_entry.setText(str(model_settings.get("top_k", 40)))
                        self.max_length_entry.setText(str(model_settings.get("max_length", 2048)))
                        
                    if "system_instructions" in session_data:
                        self.system_instructions_text.setPlainText(session_data["system_instructions"])
                        
                    Toast.show(self, f"Session loaded: {os.path.basename(file_path)}", 2000)
        except Exception as e:
            logging.error(f"File operation error: {e}")
            Toast.show(self, f"Error: {str(e)}", 3000)
    
    def load_env_file(self):
        """Load environment variables from a .env file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select .env File", "", "Env Files (.env);;All Files (*)")
            if file_path:
                load_dotenv(file_path)
                Toast.show(self, ".env file loaded successfully", 2000)
        except Exception as e:
            logging.error(f"Error loading .env file: {e}")
            Toast.show(self, f"Error loading .env file: {str(e)}", 3000)
    
    def update_goals(self):
        """Update AI goals in system instructions"""
        # This is a placeholder for updating AI goals
        goal_text, ok = QInputDialog.getMultiLineText(
            self, "AI Goals", "Define goals for the AI assistant:", 
            self.system_instructions_text.toPlainText()
        )
        
        if ok and goal_text:
            self.system_instructions_text.setPlainText(goal_text)
            Toast.show(self, "AI goals updated", 2000)
    
    def view_goals(self):
        """View current AI goals from system instructions"""
        goals = self.system_instructions_text.toPlainText()
        if not goals:
            goals = "No goals defined. Use 'Update AI Goals' to set them."
        
        # Create a non-modal dialog to show the goals
        dialog = QDialog(self)
        dialog.setWindowTitle("Current AI Goals")
        layout = QVBoxLayout(dialog)
        
        text_display = QTextEdit()
        text_display.setPlainText(goals)
        text_display.setReadOnly(True)
        layout.addWidget(text_display)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setMinimumSize(400, 300)
        dialog.show()
    
    def toggle_multi_response_mode(self, checked):
        """Toggle multi-response mode UI elements"""
        self.response_count_label.setVisible(checked)
        self.response_count_combo.setVisible(checked)
    
    def update_agent_roles(self, count):
        """Update the UI to show role inputs for the specified number of agents"""
        # Clear existing role entries
        while self.agent_roles_layout.count():
            item = self.agent_roles_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear agent role entries list
        self.agent_role_entries.clear()
        
        # Add new role entries
        for i in range(count):
            role_layout = QHBoxLayout()
            role_layout.addWidget(QLabel(f"Agent {i+1} Role:"))
            
            role_entry = QLineEdit()
            if i == 0:
                role_entry.setText("Critical thinker analyzing the problem in detail")
            elif i == 1:
                role_entry.setText("Creative problem solver offering solutions")
            else:
                role_entry.setText(f"Agent {i+1} offering unique perspective")
            
            role_layout.addWidget(role_entry)
            self.agent_roles_layout.addLayout(role_layout)
            self.agent_role_entries.append(role_entry)
    
    def toggle_agent_mode(self, enabled):
        """Toggle agent mode on/off"""
        self.agent_enabled = enabled
        Toast.show(self, f"Agent mode {'enabled' if enabled else 'disabled'}", 1500)
    
    def load_agent_preset(self):
        """Load predefined agent presets"""
        preset = self.agent_preset_combo.currentText()
        
        # Define presets
        if preset == "Debate (Pro & Con)":
            self.agent_count_spinner.setValue(2)
            self.agent_role_entries[0].setText("Presents arguments in favor of the topic, citing evidence and reasoning")
            self.agent_role_entries[1].setText("Presents arguments against the topic, citing evidence and reasoning")
            
        elif preset == "Code Review (Developer & Reviewer)":
            self.agent_count_spinner.setValue(2)
            self.agent_role_entries[0].setText("Software developer explaining code functionality and design decisions")
            self.agent_role_entries[1].setText("Code reviewer finding potential issues and suggesting improvements")
            
        elif preset == "Creative (Writer & Editor)":
            self.agent_count_spinner.setValue(2)
            self.agent_role_entries[0].setText("Creative writer generating original content and ideas")
            self.agent_role_entries[1].setText("Editor refining content for clarity, style, and coherence")
            
        elif preset == "Problem Solving (Analyst & Critic)":
            self.agent_count_spinner.setValue(3)
            self.agent_role_entries[0].setText("Analyst breaking down the problem into component parts")
            self.agent_role_entries[1].setText("Solutions expert proposing detailed solutions")
            self.agent_role_entries[2].setText("Critic evaluating solutions and identifying weaknesses")
        
        Toast.show(self, f"{preset} preset loaded", 1500)
    
    def save_agent_preset(self):
        """Save current agent configuration as a custom preset"""
        preset_name = self.preset_name_entry.text().strip()
        if not preset_name:
            Toast.show(self, "Please enter a preset name", 1500)
            return
        
        # Create a preset with current settings
        agent_roles = {}
        for i, entry in enumerate(self.agent_role_entries):
            agent_roles[i] = entry.text()
        
        # Save preset - in a real app, this would save to a file or database
        # For now, just add to the dropdown
        if preset_name not in [self.agent_preset_combo.itemText(i) for i in range(self.agent_preset_combo.count())]:
            self.agent_preset_combo.addItem(preset_name)
        
        self.agent_preset_combo.setCurrentText(preset_name)
        Toast.show(self, f"Custom preset '{preset_name}' saved", 1500)
    
    def change_theme(self, theme_name):
        """Change the application theme"""
        global current_theme
        if theme_name in themes:
            current_theme = theme_name
            self.apply_theme()
            Toast.show(self, f"Theme changed to {theme_name}", 1500)
    
    def change_font(self, element_type):
        """Change font for a specific UI element"""
        global current_fonts
        font, ok = QFontDialog.getFont(current_fonts[element_type], self)
        if ok:
            current_fonts[element_type] = font
            if element_type == "input":
                for input_entry in self.input_entries:
                    input_entry.setFont(font)
            elif element_type == "output":
                for output_text in self.output_texts:
                    output_text.setFont(font)
            elif element_type == "ui":
                # Apply UI font to appropriate elements
                pass
            
            Toast.show(self, f"{element_type.capitalize()} font updated", 1500)
    
    def change_accent_color(self):
        """Change the accent color"""
        global themes, current_theme
        color = QColorDialog.getColor(QColor(themes[current_theme]["accent"]), self, "Choose Accent Color")
        if color.isValid():
            themes[current_theme]["accent"] = color.name()
            self.apply_theme()
            Toast.show(self, "Accent color updated", 1500)
    
    def change_layout(self, layout_style):
        """Change the layout style"""
        if layout_style == "Compact":
            # Apply compact layout
            self.splitter.setSizes([200, 600])
        elif layout_style == "Expanded":
            # Apply expanded layout
            self.splitter.setSizes([400, 400])
        else:  # Standard
            # Apply standard layout
            self.splitter.setSizes([300, 500])
        
        Toast.show(self, f"{layout_style} layout applied", 1500)
    
    def apply_theme(self):
        """Apply the current theme to the UI elements"""
        theme = themes[current_theme]
        
        # Main window
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {theme['background']};
                color: {theme['text']};
            }}
            QTextEdit, QLineEdit {{
                background-color: {theme['input_bg']};
                color: {theme['text']};
                border: 1px solid {theme['secondary']};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {theme['accent']};
                color: white;
                border-radius: 4px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {theme['accent'] + '80'};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme['secondary']};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {theme['input_bg']};
                color: {theme['text']};
                border: 1px solid {theme['secondary']};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px;
            }}
            QTabBar::tab:selected {{
                background-color: {theme['accent']};
                color: white;
            }}
        """)
        
        # Update progress indicator color
        if hasattr(self, 'progress_indicator'):
            self.progress_indicator.setStyleSheet(f"""
                background-color: {theme['accent']};
                border-radius: 4px;
                width: 0%;
            """)
    
    def apply_settings(self):
        """Apply model settings from the UI"""
        global model_settings
        try:
            # Get values from UI
            temperature = float(self.temperature_entry.text())
            top_p = float(self.top_p_entry.text())
            top_k = int(self.top_k_entry.text())
            max_length = int(self.max_length_entry.text())
            model = self.model_selector.currentText()
            show_thinking = self.show_thinking_checkbox.isChecked()
            
            # Validate values
            if not (0 <= temperature <= 2):
                raise ValueError("Temperature must be between 0 and 2")
            
            if not (0 <= top_p <= 1):
                raise ValueError("Top P must be between 0 and 1")
            
            if not (1 <= top_k <= 100):
                raise ValueError("Top K must be between 1 and 100")
            
            if not (1 <= max_length <= 4096):
                raise ValueError("Max Length must be between 1 and 4096")
            
            # Update settings
            model_settings["temperature"] = temperature
            model_settings["top_p"] = top_p
            model_settings["top_k"] = top_k
            model_settings["max_length"] = max_length
            model_settings["model"] = model
            model_settings["show_thinking"] = show_thinking
            
            # Show confirmation
            Toast.show(self, "Settings applied", 1500)
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Setting", str(e))
        except Exception as e:
            logging.error(f"Settings error: {e}")
            QMessageBox.warning(self, "Error", f"Failed to apply settings: {str(e)}")
    
    def save_system_instructions(self):
        """Save the system instructions to the current preset"""
        if not instructions_manager:
            Toast.show(self, "Instructions manager not available", 1500)
            return
        
        try:
            # Get the current preset name
            preset_name = self.preset_dropdown.currentText()
            instruction_text = self.system_instructions_text.toPlainText()
            
            # Update the current preset
            if instructions_manager.set_system_instruction(preset_name, instruction_text):
                Toast.show(self, f"Saved to {preset_name} preset", 1500)
            else:
                Toast.show(self, "Failed to save instructions", 1500)
        except Exception as e:
            logging.error(f"Error saving instructions: {e}")
            Toast.show(self, f"Error: {str(e)}", 3000)

    def save_custom_preset(self):
        """Save a new custom instruction preset"""
        if not instructions_manager:
            Toast.show(self, "Instructions manager not available", 1500)
            return
            
        preset_name = self.custom_preset_name.text().strip()
        if not preset_name:
            Toast.show(self, "Please enter a preset name", 1500)
            return
        
        instruction_text = self.system_instructions_text.toPlainText()
        
        # Add the new preset
        if instructions_manager.set_system_instruction(preset_name, instruction_text):
            # Update dropdown
            if self.preset_dropdown.findText(preset_name) == -1:
                self.preset_dropdown.addItem(preset_name)
            self.preset_dropdown.setCurrentText(preset_name)
            
            # Set as current
            instructions_manager.set_current_system_instruction(preset_name)
            
            Toast.show(self, f"Custom preset '{preset_name}' saved", 1500)
            self.custom_preset_name.clear()
        else:
            Toast.show(self, "Failed to save custom preset", 1500)

    def apply_preset(self):
        """Apply an instruction preset from the JSON file"""
        if not instructions_manager:
            Toast.show(self, "Instructions manager not available", 1500)
            return
            
        preset = self.preset_dropdown.currentText()
        instruction_text = instructions_manager.get_system_instruction(preset)
        
        if instruction_text:
            self.system_instructions_text.setPlainText(instruction_text)
            instructions_manager.set_current_system_instruction(preset)
            Toast.show(self, f"{preset} preset applied", 1500)
        else:
            Toast.show(self, f"Preset '{preset}' not found", 1500)

    def browse_model_path(self):
        """Open a file dialog to browse for a model path"""
        global local_model
        
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory", 
                                                   os.path.expanduser("~"))
        if model_dir:
            self.model_path_entry.setText(model_dir)
            Toast.show(self, f"Model path set to: {model_dir}", 1500)

    def change_model_path(self):
        """Change the model path and reset the model"""
        global local_model
        
        new_path = self.model_path_entry.text().strip()
        if not new_path:
            Toast.show(self, "Please enter a valid model path", 2000)
            return
        
        # Unload current model if loaded
        if local_model and local_model.initialized:
            local_model.unload()
        
        try:
            # Update model path
            local_model.model_path = new_path
            self.model_status_indicator.setStyleSheet("color: #8E8E93;")  # Gray
            self.model_status_label.setText("Model path changed - not loaded")
            
            # Check if path is valid
            valid, message = local_model.check_model_path()
            if valid:
                Toast.show(self, f"Model path changed to: {new_path}", 2000)
            else:
                QMessageBox.warning(self, "Warning", f"Model path may not be valid: {message}")
                Toast.show(self, "Model path changed but might not be valid", 2000)
        except Exception as e:
            logging.error(f"Error changing model path: {e}")
            Toast.show(self, f"Error: {str(e)}", 3000)

    def show_system_info(self):
        """Display system information in a dialog"""
        global local_model
        
        if not local_model:
            QMessageBox.warning(self, "Error", "Model handler not initialized")
            return
        
        # Get system information
        sys_info = local_model.get_system_info()
        
        # Format system info for display
        info_text = "System Information:\n\n"
        for key, value in sys_info.items():
            info_text += f"{key}: {value}\n"
        
        # Add model information
        info_text += f"\nModel Path: {local_model.model_path}\n"
        info_text += f"Model Loaded: {'Yes' if local_model.initialized else 'No'}\n"
        
        # Display in dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("System Information")
        layout = QVBoxLayout(dialog)
        
        text_display = QTextEdit()
        text_display.setPlainText(info_text)
        text_display.setReadOnly(True)
        layout.addWidget(text_display)
        
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(info_text))
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(copy_btn)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        dialog.setMinimumSize(500, 400)
        dialog.exec_()

    def toggle_offline_mode(self, enabled):
        """Toggle offline mode for model loading"""
        global local_model
        if local_model:
            local_model.set_offline_mode(enabled)
            Toast.show(self, f"Offline mode {'enabled' if enabled else 'disabled'}", 1500)
            
            if enabled:
                # Warn if using a model ID that's not in cache
                if not os.path.exists(local_model.model_path):
                    cache_path = os.path.join(local_model.cache_dir, local_model.model_path.replace("/", "--"))
                    if not os.path.exists(cache_path):
                        QMessageBox.warning(self, "Offline Mode Warning", 
                                f"Current model '{local_model.model_path}' is not available locally.\n\n"
                                f"Either:\n1. Use 'Pre-Download Model' button, or\n"
                                f"2. Select a local model directory")

    def browse_cache_dir(self):
        """Open a file dialog to browse for a cache directory"""
        global local_model
        
        cache_dir = QFileDialog.getExistingDirectory(self, "Select Cache Directory", 
                                                   os.path.expanduser("~"))
        if cache_dir:
            self.cache_dir_entry.setText(cache_dir)
            if local_model:
                local_model.set_cache_dir(cache_dir)
            Toast.show(self, f"Cache directory set to: {cache_dir}", 1500)

    def download_model(self):
        """Open a dialog to download a model for offline use"""
        global local_model
        
        # Default to current model if it's not found locally
        default_model_id = ""
        if local_model:
            current_model_is_missing = False
            if "DeepHermes" in local_model.model_path:
                cache_path = os.path.join(local_model.cache_dir, local_model.model_path.replace("/", "--"))
                current_model_is_missing = not os.path.exists(cache_path)
            
            if current_model_is_missing:
                default_model_id = local_model.model_path
        
        # Ask for model ID
        model_id, ok = QInputDialog.getText(self, "Download Model",
                            "Enter HuggingFace model ID (e.g., meta-llama/Llama-2-7b-chat):", 
                            QLineEdit.Normal, 
                            default_model_id)
        if not ok or not model_id:
            return
        
        # Confirm download
        cache_dir = self.cache_dir_entry.text()
        reply = QMessageBox.question(self, "Confirm Download",
                                    f"Download model '{model_id}' to:\n{cache_dir}?\n\n"
                                    f"This may take a long time depending on model size.",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Start download in a thread to avoid freezing UI
            self.status_left.setText(f"Downloading model: {model_id}...")
            
            download_thread = Thread(target=self._download_model_thread, 
                                    args=(model_id, cache_dir))
            download_thread.daemon = True
            download_thread.start()

    def _download_model_thread(self, model_id, cache_dir):
        """Background thread for model downloading"""
        try:
            download_model_to_cache(model_id, cache_dir)
            # Signal completion to the main thread
            QMetaObject.invokeMethod(
                self, 
                "handle_model_download_complete",
                Qt.QueuedConnection,
                Q_ARG(str, model_id),
                Q_ARG(bool, True),
                Q_ARG(str, "")
            )
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error downloading model: {error_msg}")
            QMetaObject.invokeMethod(
                self, 
                "handle_model_download_complete",
                Qt.QueuedConnection,
                Q_ARG(str, model_id),
                Q_ARG(bool, False),
                Q_ARG(str, error_msg)
            )

    @pyqtSlot(str, bool, str)
    def handle_model_download_complete(self, model_id, success, error_msg):
        """Handle completion of model download"""
        if success:
            self.status_left.setText("Ready")
            
            # If we had a model path set, offer to switch to offline mode
            if local_model and model_id == local_model.model_path and not local_model.offline_mode:
                reply = QMessageBox.question(self, "Model Downloaded", 
                                            f"Model '{model_id}' was successfully downloaded for offline use.\n\n"
                                            f"Would you like to enable offline mode now?",
                                            QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.offline_mode_check.setChecked(True)
                    local_model.set_offline_mode(True)
            else:
                QMessageBox.information(self, "Download Complete", 
                                      f"Model '{model_id}' was successfully downloaded for offline use.")
        else:
            self.status_left.setText("Ready")
            
            # Special handling for NousResearch/DeepHermes models
            if "DeepHermes" in model_id:
                error_dialog = QMessageBox(self)
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Download Failed")
                error_dialog.setText(f"Failed to download DeepHermes model")
                error_dialog.setInformativeText(
                    f"This model requires HuggingFace authentication.\n\n"
                    f"Would you like to run the specialized DeepHermes setup script?"
                )
                error_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                response = error_dialog.exec_()
                
                if response == QMessageBox.Yes:
                    self.run_deephermes_setup()
            else:
                QMessageBox.critical(self, "Download Failed", 
                                   f"Failed to download model '{model_id}':\n\n{error_msg}")

    def change_gpu(self):
        """Change the GPU used for inference"""
        global local_model
        
        if local_model and torch.cuda.is_available():
            selected_idx = self.gpu_selector.currentIndex()
            gpu_id = self.gpu_selector.itemData(selected_idx)
            
            if gpu_id >= 0:
                # Check if currently loaded model needs to be unloaded
                if local_model.initialized:
                    reply = QMessageBox.question(self, "GPU Change", 
                                               "Changing GPU requires unloading the current model. Continue?",
                                               QMessageBox.Yes | QMessageBox.No)
                    
                    if reply == QMessageBox.Yes:
                        local_model.unload()
                    else:
                        return
                
                # Set the new GPU
                success, message = local_model.set_gpu_device(gpu_id)
                
                if success:
                    Toast.show(self, f"GPU changed: {message}", 2000)
                    self.update_memory_usage()
                else:
                    QMessageBox.warning(self, "GPU Error", message)
            else:
                QMessageBox.warning(self, "GPU Error", "No valid GPU selected")
        else:
            QMessageBox.warning(self, "GPU Error", "CUDA not available or model not initialized")

# ------------------------------------------------------------------------------
# Main application entry point
def main():
    # Set application attributes
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    truehalt_dir = os.path.join(os.path.expanduser("~"), "TrueHalt")
    os.makedirs(truehalt_dir, exist_ok=True)
    
    # Set the correct path for system instructions
    instructions_path = os.path.join(truehalt_dir, "system_instructions.json")
    # Create application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a consistent look across platforms
    
    # Check for required packages
    try:
        import bitsandbytes
        logging.info("bitsandbytes library found (required for 4-bit quantization)")
    except ImportError:
        logging.warning("bitsandbytes library not found - 4-bit quantization will not work")
        warning_dialog = QMessageBox()
        warning_dialog.setIcon(QMessageBox.Warning)
        warning_dialog.setText("Missing required library: bitsandbytes")
        warning_dialog.setInformativeText("This library is needed for 4-bit model quantization.\n\n"
                                         "Install it with: pip install bitsandbytes")
        warning_dialog.setWindowTitle("Library Missing")
        warning_dialog.setStandardButtons(QMessageBox.Ok)
        warning_dialog.exec_()
    
    # Create and show main window
    window = TrueHaltApp()
    window.show()
    
    # Force CUDA optimizations
    optimize_gpu_settings()
    
    # Try to force RTX 3070 Ti if available
    if torch.cuda.is_available():
        # Look for RTX 3070 Ti
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            if "3070 Ti" in gpu_name:
                torch.cuda.set_device(i)
                logging.info(f"Forced use of RTX 3070 Ti (Device {i})")
                if local_model:
                    local_model.gpu_id = i
                    local_model.device = f"cuda:{i}"
                break
    
    # Run application event loop
    sys.exit(app.exec_())

# Run the application if this script is executed directly
if __name__ == "__main__":
    main()