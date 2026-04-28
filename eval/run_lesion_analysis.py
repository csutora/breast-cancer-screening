"""Command-line entrypoint for lesion feature analysis and plotting."""

from __future__ import annotations

import argparse

from lesion_analysis import AnalysisInputs, run_full_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract lesion features (size, position, boundary clarity) and generate plots."
    )
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument(
        "--train_csv",
        default="./meta/cbis-ddsm/mass_case_description_train_set.csv",
    )
    parser.add_argument(
        "--test_csv",
        default="./meta/cbis-ddsm/mass_case_description_test_set.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="./outputs/lesion_features.csv",
        help="Where to save the lesion-level feature table.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/plots",
        help="Where to save histogram and position plot images.",
    )
    args = parser.parse_args()

    inputs = AnalysisInputs(
        data_root=args.data_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
    )

    df, plot_paths = run_full_analysis(
        inputs=inputs,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
    )

    print(f"[analysis] lesions extracted: {len(df)}")
    print(f"[analysis] feature table: {args.output_csv}")
    print("[analysis] generated plots:")
    for p in plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
