#!/bin/bash

# Define the target patterns
patterns=("route_Town*.xml.gz" "scenario_counts_*.json")

# Prevent the loop from running if no files match
shopt -s nullglob

for pattern in "${patterns[@]}"; do
    for file in $pattern; do

        # This regex captures:
        # 1. Everything before the last underscore (prefix)
        # 2. The digits after that underscore (the ID)
        # 3. Everything after those digits (the extension)
        if [[ $file =~ ^(.*_)([0-9]+)(\..*)$ ]]; then
            prefix="${BASH_REMATCH[1]}"
            id_str="${BASH_REMATCH[2]}"
            suffix="${BASH_REMATCH[3]}"

            # Force base 10 to avoid octal errors
            id_val=$((10#$id_str))

            # Apply the mapping (adding 64)
            new_id_val=$((id_val + 64))

            # Format to at least 2 digits (e.g., 0 -> 64, 63 -> 127)
            new_id_str=$(printf "%02d" $new_id_val)

            new_filename="${prefix}${new_id_str}${suffix}"

            # Safety: check if destination already exists to avoid overwriting
            if [[ -e "$new_filename" ]]; then
                echo "SKIPPING: $new_filename already exists!"
            else
                echo "Renaming: $file -> $new_filename"
                mv "$file" "$new_filename"
            fi
        fi
    done
done

echo "Process complete."