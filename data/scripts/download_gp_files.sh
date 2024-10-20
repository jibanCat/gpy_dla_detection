#!/bin/bash

# Define the GitHub repository and the target folder
REPO_URL="https://github.com/jibanCat/gp_dr12_trained.git"
TARGET_FOLDER="data/dr12q/processed"

# Clone the repository
echo "Cloning the repository..."
git clone "$REPO_URL" temp_repo

# Check if the clone was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone the repository."
    exit 1
fi

# Create the target folder if it doesn't exist
echo "Creating target folder: $TARGET_FOLDER"
mkdir -p "$TARGET_FOLDER"

# Move the files from the cloned repository to the target folder
echo "Moving files to the target folder..."
mv temp_repo/* "$TARGET_FOLDER"

# Remove the temporary repository folder
echo "Cleaning up..."
rm -rf temp_repo

echo "Download and setup complete."