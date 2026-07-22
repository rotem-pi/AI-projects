#!/usr/bin/env bash
# Sets up and launches definiData with minimal effort:
#   ./init.sh
set -e

cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "python3 is required. Install it and re-run this script."
    exit 1
fi

if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Checking AWS credentials..."
if ! python3 -c "import boto3; boto3.client('sts', region_name='eu-north-1').get_caller_identity()" &> /dev/null; then
    echo ""
    echo "No valid AWS credentials found."
    echo "Run 'aws login' (or 'aws sso login' if your org uses that) to sign in, then re-run this script."
    exit 1
fi

# Skip Streamlit's interactive first-run "email address" prompt, which would
# otherwise hang with no stdin available.
mkdir -p ~/.streamlit
if [ ! -f ~/.streamlit/credentials.toml ]; then
    printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
fi

echo "Starting definiData..."
streamlit run app.py
