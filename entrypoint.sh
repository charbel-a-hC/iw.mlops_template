#!/bin/bash

# CD into repo
cd ${REPO_NAME}

# Set up the git safe directory
git config --global --add safe.directory /${REPO_NAME}

# Append the Python path with the repository directory
export PYTHONPATH=${PWD}/${REPO_NAME}:$PYTHONPATH

# Check if the user provided a Python script to execute
if [ $# -eq 0 ]; then
  echo "Please provide a Python script as an argument."
  exit 1
fi

# Run Python3 with the provided script
python3 "$@"