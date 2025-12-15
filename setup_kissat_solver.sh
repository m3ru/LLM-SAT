#!/bin/bash
# Setup Kissat SAT solver base template
# This script downloads and configures the Kissat solver that will be used as a template
# for generating modified solvers with LLM-generated restart policies.

set -euo pipefail

echo "Setting up Kissat solver base template..."

# Create solvers directory if it doesn't exist
mkdir -p solvers

# Remove old base if it exists
if [ -d "solvers/base" ]; then
    echo "Removing old solvers/base directory..."
    rm -rf solvers/base
fi

# Clone Kissat repository (official version)
echo "Cloning Kissat repository..."
git clone https://github.com/arminbiere/kissat.git solvers/base

cd solvers/base
./configure
