"""
FILE: main.py
PURPOSE: Run full pipeline with config, reporting, and automatic cleanup
"""

import argparse
import json
from pathlib import Path

from data.generate_synthetic_data import generate_data
from src.train import train
from src.evaluate import evaluate
from src.ml_models import train_all_models
from src.visualize import (
    plot_learning_curve,
    plot_step_by_step,
    plot_evaluation_comparison,
    plot_ml_comparison,
)
from src.config import load_config, print_config_summary
from src.report import generate_report


def cleanup_reports(keep_last=1):
    """Delete old report files, keeping only the most recent ones."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return

    print("\n🧹 Cleaning up old reports...")

    # Get all report files
    report_files = list(reports_dir.glob("report_*.txt"))
    csv_files = list(reports_dir.glob("*.csv"))
    all_files = report_files + csv_files

    if not all_files:
        print("   No old reports to clean")
        return

    # Sort by creation time (newest first)
    all_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)

    # Keep the newest 'keep_last' files, delete the rest
    files_to_delete = all_files[keep_last:]

    for f in files_to_delete:
        try:
            f.unlink()
            print(f"   Deleted: {f.name}")
        except Exception as e:
            print(f"   Could not delete {f.name}: {e}")

    print(f"   Kept {len(all_files[:keep_last])} most recent files")


def main():
    parser = argparse.ArgumentParser(description="Traffic RL Project")

    # Basic arguments
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override number of episodes"
    )
    parser.add_argument("--skip-data", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML model training")
    parser.add_argument(
        "--report", action="store_true", help="Generate report after run"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test (500 episodes, skip ML)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override episodes if provided
    if args.episodes:
        config["training"]["n_episodes"] = args.episodes

    # Quick test mode
    if args.quick_test:
        print("\n🔧 QUICK TEST MODE - 500 episodes, skipping ML")
        config["training"]["n_episodes"] = 500
        args.skip_ml = True
        args.report = True  # Auto-generate report for quick test

    # Print configuration summary
    print_config_summary(config)

    print("\n" + "=" * 60)
    print("  TRAFFIC SIGNAL OPTIMIZATION WITH Q-LEARNING")
    print("=" * 60)

    # Step 1: Generate Data
    if not args.skip_data:
        print("\n[1/6] Generating synthetic data...")
        generate_data(
            n_days=config["data"]["n_days"],
            n_intersections=config["data"]["n_intersections"],
        )
    else:
        print("\n[1/6] Skipping data generation")

    # Step 2: Train ML Models
    if not args.skip_ml:
        print("\n[2/6] Training ML models...")
        ml_results = train_all_models()
        # Save ML results
        Path("logs").mkdir(exist_ok=True)
        with open("logs/ml_metrics.json", "w") as f:
            json.dump(ml_results, f, indent=2)
    else:
        print("\n[2/6] Skipping ML training")
        ml_results = {
            "RandomForest": {"mae": 0, "r2": 0},
            "LinearRegression": {"mae": 0, "r2": 0},
            "SVR": {"mae": 0, "r2": 0},
        }

    # Step 3: Train RL Agent
    print("\n[3/6] Training Q-Learning agent...")
    history = train(
        n_episodes=config["training"]["n_episodes"],
        max_steps=config["environment"]["max_steps_per_episode"],
    )

    # Step 4: Evaluate
    print("\n[4/6] Evaluating...")
    eval_results = evaluate(n_episodes=config["evaluation"]["n_episodes"])

    # Save evaluation results
    with open("logs/evaluation_results.json", "w") as f:
        # Convert numpy values to Python floats
        clean_results = {}
        for agent, metrics in eval_results.items():
            clean_results[agent] = {
                "mean_reward": float(metrics["mean_reward"]),
                "std_reward": float(metrics["std_reward"]),
                "mean_queue": float(metrics["mean_queue"]),
                "std_queue": float(metrics["std_queue"]),
            }
        json.dump(clean_results, f, indent=2)

    # Step 5: Create Charts
    print("\n[5/6] Creating charts...")
    plot_learning_curve()
    plot_step_by_step()
    plot_evaluation_comparison(eval_results)
    if not args.skip_ml:
        plot_ml_comparison(ml_results)

    # Step 6: Generate Report (if requested)
    if args.report:
        # Clean up old reports first (keep only last 1)
        cleanup_reports(keep_last=1)

        print("\n" + "=" * 60)
        print("[6/6] Generating report...")
        print("=" * 60)

        try:
            print("   Calling generate_report()...")
            result = generate_report()

            if result:
                print(f"\n   ✅ Report generated successfully!")
                print(f"   📄 Text report: {result['text_report']}")
                print(f"   📊 CSV files: {', '.join(result['csv_files'])}")

                # Show preview of report
                print("\n   📋 Report Preview (first 10 lines):")
                print("   " + "-" * 40)
                with open(result["text_report"], "r") as f:
                    preview = f.readlines()[:10]
                    for line in preview:
                        print(f"      {line.rstrip()}")
                print("   " + "-" * 40)
            else:
                print("   ❌ Report generation failed - no result returned")

        except Exception as e:
            print(f"   ❌ Report generation failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print("   📊 Charts: results/ folder")
    print("   🤖 Model: models/q_table.npy")
    print("   📈 Logs: logs/ folder")
    if args.report:
        print("   📄 Report: reports/ folder")
        # Show the latest report file
        reports_dir = Path("reports")
        if reports_dir.exists():
            latest_reports = list(reports_dir.glob("report_*.txt"))
            if latest_reports:
                latest = max(latest_reports, key=lambda x: x.stat().st_ctime)
                print(f"      Latest: {latest.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
