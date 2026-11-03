#!/bin/bash
# One-time: gh auth login --web   (choose the MDavis3 account; complete the passkey prompt)
set -e
cd "$(dirname "$0")/.."
gh auth switch -u MDavis3
gh repo create MDavis3/platonic-brain-index --private --source . --remote origin --push \
  --description "Platonic Brain Index: do brains and AI models converge on one representation? Preregistered, reproducible benchmark on THINGS EEG/MEG/fMRI."
echo "Pushed. Make it public when ready:  gh repo edit MDavis3/platonic-brain-index --visibility public --accept-visibility-change-consequences"
