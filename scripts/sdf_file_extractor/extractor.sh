#!/bin/bash

CSV="/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/new_data_pred/top_scorers/top_0.1_percent.csv"
SRC="data/minimized_output"
DEST="sdf_top_1_pct"
MISSING="missing_sdfs.txt"

mkdir -p "$DEST"
> "$MISSING"   # Empty the missing file

# Skip header and read the identifier column (assumes it's the first column)
tail -n +2 "$CSV" | cut -d',' -f1 | while read -r id
do
    file="minimized_${id}.sdf"

    if [ -f "$SRC/$file" ]; then
        cp "$SRC/$file" "$DEST/"
    else
        echo "$file" >> "$MISSING"
    fi
done

echo "Done."