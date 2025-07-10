#!/bin/bash
set -e

# Define colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check GPU availability
check_gpu() {
    echo -e "${BLUE}[INFO]${NC} Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        if [ $NUM_GPUS -gt 0 ]; then
            echo -e "${GREEN}[SUCCESS]${NC} Found ${NUM_GPUS} GPU(s) available for training."
            return 0
        else
            echo -e "${YELLOW}[WARNING]${NC} No GPUs found despite nvidia-smi being available."
            return 1
        fi
    else
        echo -e "${YELLOW}[WARNING]${NC} nvidia-smi not found. Running in CPU-only mode."
        return 1
    fi
}

# Function to print system information
print_system_info() {
    echo -e "${BLUE}[INFO]${NC} System Information:"
    echo -e "  - Python version: $(python --version 2>&1)"
    echo -e "  - PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo -e "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    if python -c 'import torch; print(torch.cuda.is_available())' | grep -q 'True'; then
        echo -e "  - CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
        echo -e "  - GPU(s): $(python -c 'import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])')"
    fi
    echo -e "  - CPU cores: $(grep -c ^processor /proc/cpuinfo)"
    echo -e "  - Memory: $(free -h | grep Mem | awk '{print $2}')"
}

# Function to print welcome message
print_welcome() {
    echo -e "\n${GREEN}=======================================================${NC}"
    echo -e "${GREEN}   Polar-RTDETRv2 Training Environment${NC}"
    echo -e "${GREEN}=======================================================${NC}"
    echo -e "${BLUE}A real-time face detection model with landmark prediction${NC}"
    echo -e "${BLUE}Optimized for WiderFace dataset with 5 landmarks${NC}"
    echo -e "${GREEN}=======================================================${NC}\n"
}

# Function to print help message
print_help() {
    echo -e "\n${BLUE}Usage:${NC}"
    echo -e "  ./entrypoint.sh [command] [options]"
    echo -e "\n${BLUE}Commands:${NC}"
    echo -e "  train                Start training with specified options"
    echo -e "  eval                 Evaluate model on validation set"
    echo -e "  export               Export model to ONNX format"
    echo -e "  demo                 Run inference demo"
    echo -e "  shell                Start interactive shell"
    echo -e "  help                 Show this help message"
    echo -e "\n${BLUE}Options:${NC}"
    echo -e "  --config CONFIG      Path to configuration file"
    echo -e "  --batch-size SIZE    Batch size for training"
    echo -e "  --epochs EPOCHS      Number of epochs to train"
    echo -e "  --workers WORKERS    Number of data loading workers"
    echo -e "  --resume PATH        Path to checkpoint to resume from"
    echo -e "  --gpu-ids IDS        GPU IDs to use (comma-separated)"
    echo -e "  --data-path PATH     Path to dataset"
    echo -e "  --output-dir DIR     Output directory for logs and checkpoints"
    echo -e "\n${BLUE}Examples:${NC}"
    echo -e "  ./entrypoint.sh train --config configs/widerface.yaml --batch-size 16"
    echo -e "  ./entrypoint.sh eval --config configs/widerface.yaml --resume checkpoints/model_best.pth"
    echo -e "  ./entrypoint.sh shell"
    echo -e "${GREEN}=======================================================${NC}\n"
}

# Function to check and create directories
setup_directories() {
    echo -e "${BLUE}[INFO]${NC} Setting up directories..."
    
    # Create necessary directories if they don't exist
    mkdir -p /app/data/widerface
    mkdir -p /app/configs
    mkdir -p /app/outputs/logs
    mkdir -p /app/outputs/checkpoints
    mkdir -p /app/outputs/visualizations
    
    echo -e "${GREEN}[SUCCESS]${NC} Directories setup complete."
}

# Function to check for dataset
check_dataset() {
    echo -e "${BLUE}[INFO]${NC} Checking for WiderFace dataset..."
    
    if [ -d "/app/data/widerface/train" ] && [ -d "/app/data/widerface/val" ]; then
        echo -e "${GREEN}[SUCCESS]${NC} WiderFace dataset found."
    else
        echo -e "${YELLOW}[WARNING]${NC} WiderFace dataset not found in expected location."
        echo -e "${YELLOW}[WARNING]${NC} Please ensure your dataset is mounted at /app/data/widerface with train and val subdirectories."
    fi
}

# Main execution
main() {
    print_welcome
    setup_directories
    check_gpu
    print_system_info
    check_dataset
    
    # Parse command line arguments
    COMMAND=${1:-help}
    shift 2>/dev/null || true
    
    case $COMMAND in
        train)
            echo -e "${BLUE}[INFO]${NC} Starting training with arguments: $@"
            python -m tools.train $@
            ;;
        eval)
            echo -e "${BLUE}[INFO]${NC} Starting evaluation with arguments: $@"
            python -m tools.eval $@
            ;;
        export)
            echo -e "${BLUE}[INFO]${NC} Exporting model with arguments: $@"
            python -m tools.export $@
            ;;
        demo)
            echo -e "${BLUE}[INFO]${NC} Running demo with arguments: $@"
            python -m tools.demo $@
            ;;
        shell)
            echo -e "${BLUE}[INFO]${NC} Starting interactive shell"
            /bin/bash
            ;;
        help|*)
            print_help
            ;;
    esac
}

# Execute main function
main "$@"
