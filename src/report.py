import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.config_loader import get_section


class ReportGenerator:
    """Generate reports from training logs and evaluation results."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_config = get_section("reporting", {})
        Path("reports").mkdir(exist_ok=True)
        print(f"   ReportGenerator initialized, timestamp: {self.timestamp}")

    def load_data(self):
        """Load all results from logs."""
        print("   Loading data from logs...")

        # Load training history
        history_path = Path("logs/training_history.json")
        if history_path.exists():
            with open(history_path, "r") as f:
                self.history = json.load(f)
            print(
                f"   ✓ Loaded training history: {len(self.history.get('episode_rewards', []))} episodes"
            )
        else:
            self.history = None
            print("   ⚠ No training history found")

        # Load evaluation results
        eval_path = Path("logs/evaluation_results.json")
        if eval_path.exists():
            with open(eval_path, "r") as f:
                self.eval_results = json.load(f)
            print(f"   ✓ Loaded evaluation results")
        else:
            self.eval_results = None
            print("   ⚠ No evaluation results found")

        # Load ML metrics
        ml_path = Path("logs/ml_metrics.json")
        if ml_path.exists():
            with open(ml_path, "r") as f:
                self.ml_results = json.load(f)
            print(f"   ✓ Loaded ML metrics")
        else:
            self.ml_results = None

    def calculate_metrics(self):
        """Calculate key performance metrics."""
        metrics = {}

        if self.history:
            rewards = self.history["episode_rewards"]
            queues = self.history["mean_queue_per_ep"]

            metrics["training"] = {
                "final_reward": float(np.mean(rewards[-50:])),
                "final_queue": float(np.mean(queues[-50:])),
                "best_reward": float(max(rewards)),
                "best_queue": float(min(queues)),
                "total_episodes": len(rewards),
            }
            print(f"   ✓ Calculated training metrics")

        if self.eval_results:
            fixed_queue = self.eval_results["Fixed Timer"]["mean_queue"]
            rl_queue = self.eval_results["RL Agent"]["mean_queue"]
            random_queue = self.eval_results["Random Agent"]["mean_queue"]

            metrics["evaluation"] = {
                "rl_queue": rl_queue,
                "fixed_queue": fixed_queue,
                "random_queue": random_queue,
                "improvement_vs_fixed": ((fixed_queue - rl_queue) / fixed_queue * 100),
                "improvement_vs_random": (
                    (random_queue - rl_queue) / random_queue * 100
                ),
            }
            print(f"   ✓ Calculated evaluation metrics")

        self.metrics = metrics
        return metrics

    def generate_text_report(self):
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("TRAFFIC SIGNAL RL PROJECT - FINAL REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        if "training" in self.metrics:
            t = self.metrics["training"]
            lines.append("\nTRAINING SUMMARY:")
            lines.append(f"  Episodes: {t['total_episodes']}")
            lines.append(f"  Final avg reward: {t['final_reward']:.3f}")
            lines.append(f"  Final avg queue: {t['final_queue']:.2f} cars")
            lines.append(f"  Best reward: {t['best_reward']:.3f}")
            lines.append(f"  Best queue: {t['best_queue']:.2f} cars")

        if "evaluation" in self.metrics:
            e = self.metrics["evaluation"]
            lines.append("\nEVALUATION RESULTS:")
            lines.append(f"  RL Agent queue: {e['rl_queue']:.2f} cars")
            lines.append(f"  Fixed Timer queue: {e['fixed_queue']:.2f} cars")
            lines.append(f"  Random Agent queue: {e['random_queue']:.2f} cars")
            lines.append(
                f"\n  Improvement vs Fixed Timer: {e['improvement_vs_fixed']:.1f}%"
            )
            lines.append(f"  Improvement vs Random: {e['improvement_vs_random']:.1f}%")

        if self.ml_results:
            lines.append("\nML MODEL COMPARISON:")
            for model, metrics in self.ml_results.items():
                lines.append(
                    f"  {model}: MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def generate_csv_report(self):
        """Generate CSV with all metrics."""
        csv_files = []

        if self.history:
            # Training metrics CSV
            data = []
            for i, (r, q, eps) in enumerate(
                zip(
                    self.history["episode_rewards"],
                    self.history["mean_queue_per_ep"],
                    self.history["epsilon_values"],
                )
            ):
                data.append({"episode": i, "reward": r, "queue": q, "epsilon": eps})

            df = pd.DataFrame(data)
            csv_path = f"reports/training_metrics_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            csv_files.append(csv_path)
            print(f"   ✓ Saved: {csv_path}")

        # Summary CSV
        if "evaluation" in self.metrics:
            e = self.metrics["evaluation"]
            summary = pd.DataFrame(
                [
                    {"metric": "RL Queue (cars)", "value": f"{e['rl_queue']:.2f}"},
                    {
                        "metric": "Fixed Queue (cars)",
                        "value": f"{e['fixed_queue']:.2f}",
                    },
                    {
                        "metric": "Random Queue (cars)",
                        "value": f"{e['random_queue']:.2f}",
                    },
                    {
                        "metric": "Improvement vs Fixed (%)",
                        "value": f"{e['improvement_vs_fixed']:.1f}",
                    },
                    {
                        "metric": "Improvement vs Random (%)",
                        "value": f"{e['improvement_vs_random']:.1f}",
                    },
                ]
            )
            summary_path = f"reports/summary_{self.timestamp}.csv"
            summary.to_csv(summary_path, index=False)
            csv_files.append(summary_path)
            print(f"   ✓ Saved: {summary_path}")

        return csv_files

    def generate(self):
        """Generate all reports."""
        print("\n" + "=" * 60)
        print("GENERATING REPORTS")
        print("=" * 60)

        self.load_data()

        if not self.history and not self.eval_results:
            print("   ❌ No data found in logs/ directory")
            return None

        self.calculate_metrics()

        # Text report
        text_report = self.generate_text_report()
        report_path = f"reports/report_{self.timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(text_report)
        print(f"   ✓ Text report: {report_path}")

        # CSV reports
        csv_files = self.generate_csv_report()

        # Print summary to console
        print("\n" + text_report)

        return {"text_report": report_path, "csv_files": csv_files}


def generate_report():
    """Convenience function to generate report."""
    print("   Starting report generation...")
    generator = ReportGenerator()
    result = generator.generate()
    if result:
        print(f"   ✓ Report generation complete. Files saved to reports/")
    else:
        print("   ❌ Report generation failed")
    return result


if __name__ == "__main__":
    generate_report()
