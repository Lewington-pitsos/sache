# if no argument is passed, raise an error

if [ $# -eq 0 ]; then
    echo "Error: No argument passed. Please pass a message to be encrypted."
    exit 1
fi

MSG=$1


current_version=$(grep -E '^version = "[0-9]+\.[0-9]+\.[0-9]+"' pyproject.toml | sed -E 's/version = "(.*)"/\1/')

# Split the version into its components
IFS='.' read -r major minor patch <<< "$current_version"

# Increment the patch version by 1
new_patch=$((patch + 1))

# Create the new version string
new_version="$major.$minor.$new_patch"

# Update the version in pyproject.toml
sed -i '' "s/version = \"$current_version\"/version = \"$new_version\"/" pyproject.toml

# Print the new version
echo "Updated pyproject version: $new_version"


gcn $MSG