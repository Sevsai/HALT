# HALT
Halt is a desktop application for interacting with local Large Language Models (LLMs). It provides an intuitive interface for running inference on locally downloaded models with support for optimizations like 4-bit quantization.

## Features

- **Local LLM Support**: Run models locally without internet connectivity
- **Model Management**: Download, cache, and manage large language models
- **Multi-Agent Mode**: Create conversations between multiple AI agents with different roles
- **System Instructions**: Customize AI behavior with predefined or custom instructions
- **GPU Acceleration**: Optimized for CUDA with GPU selection and memory management
- **Session Management**: Save and load conversation sessions
- **Customizable UI**: Change themes, fonts, and layout

## Requirements

- Windows or Linux with Python 3.8+
- NVIDIA GPU with CUDA support (min 8GB VRAM recommended)
- At least 16GB system RAM

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python HALT2.py
   ```

## Quick Start

1. **Download a Model**:
   - Go to the "Tools" tab
   - Click "Pre-Download Model" and enter a HuggingFace model ID
   - For example: `NousResearch/Hermes-2-Pro-Mistral-7B`

2. **Enable Offline Mode**:
   - Check "Offline Mode" on the Tools tab

3. **Check the Model**:
   - Go back to the "Chat" tab
   - Click "Check Model" to load the model

4. **Start Chatting**:
   - Type a message in the input area
   - Click "Generate Response"

## Configuration

### Model Settings

- **Temperature**: Controls randomness (0.0-1.0)
- **Top-K/Top-P**: Sampling parameters for generation
- **Max Length**: Maximum token length for responses

### System Instructions

The application uses a system instructions manager to provide context to the AI. You can:
- Select from predefined instruction presets
- Create custom instruction presets
- Save and load instruction configurations

## Multi-Agent Mode

TrueHalt supports conversations between multiple AI agents:

1. Go to the "Agents" tab
2. Enable "Multi-Agent Mode"
3. Configure number of agents and roles
4. Return to the chat and enter a prompt
5. Agents will discuss the topic with their assigned perspectives

## Advanced Usage

### Model Downloading

For manual model downloads:
```
python model_downloader.py --model MODEL_ID --dir OUTPUT_DIR
```

### DeepHermes Setup

For specific models like DeepHermes:
```
python deephermes_setup.py
```

## Troubleshooting

- **Out of Memory Errors**: Try a smaller model or enable 4-bit quantization
- **Model Not Found**: Check that the model path is correct and the model is downloaded
- **CUDA Errors**: Ensure you have compatible NVIDIA drivers installed
- **Adding Models**: Ensure you have added the model id of any model downloaded to the HALT.py file
## License

This software is provided for educational and research purposes only.
