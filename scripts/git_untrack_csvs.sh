#!/bin/sh
# Untrack CSV files so .gitignore can take effect.
# Run from project root: bash scripts/git_untrack_csvs.sh

set -e

echo "1/3: Showing currently tracked CSVs (if any):"
git ls-files -- '*.csv' || true

echo
echo "2/3: Removing CSVs from the git index (keeps files locally)..."
# untrack CSVs under data/ first (safer), then any remaining tracked csvs
git rm --cached -r --ignore-unmatch data/*.csv || true
git rm --cached -r --ignore-unmatch "*.csv" || true

echo
echo "3/3: Done. Now commit the change to stop tracking CSVs:"
echo "   git add .gitignore"
echo "   git commit -m \"Stop tracking generated CSVs; update .gitignore\""
echo "   git push"
echo
git status --porcelain
