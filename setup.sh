#!/bin/bash
# Setup script for FL-70% Lightweight 3D U-Net project

set -e  # Exit on error

echo "========================================="
echo "FL-70% Lightweight 3D U-Net Setup"
echo "========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/splits
mkdir -p models/checkpoints
mkdir -p logs/tensorboard
mkdir -p inference/prob_maps
mkdir -p inference/bboxes
mkdir -p configs

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate  # On Linux/Mac"
echo "   venv\\Scripts\\activate    # On Windows"
echo ""
echo "2. Place your FL data in data/raw/ directory"
echo ""
echo "3. Run the pipeline:"
echo "   python main.py --mode all"
echo ""
echo "For more information, see README.md"
