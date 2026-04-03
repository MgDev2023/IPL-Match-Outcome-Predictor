"""
One-shot setup script.
Run this after placing matches.csv in data/raw/

Usage:
    python setup_and_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    raw_path = Path("data/raw/matches.csv")
    if not raw_path.exists():
        print("=" * 60)
        print("ERROR: data/raw/matches.csv not found!")
        print()
        print("Download the dataset from Kaggle:")
        print("  Search: 'IPL Complete Dataset 2008-2024' on kaggle.com")
        print("  Direct dataset: kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020")
        print()
        print("Place matches.csv inside:  data/raw/matches.csv")
        print("=" * 60)
        sys.exit(1)

    print("Step 1/2 — Feature Engineering")
    from feature_engineering import run as fe_run
    fe_run()

    print("\nStep 2/2 — Training Models")
    from train_model import run as train_run
    train_run()

    print("\n" + "=" * 60)
    print("Setup complete! Launch the app with:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
