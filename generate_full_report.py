"""
FILE: generate_full_report.py
PURPOSE: Generate a complete PDF report with all code and outputs
FIXED: Unicode error with emojis
"""

import os
import json
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Try to import optional dependencies
try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except:
    FPDF_AVAILABLE = False
    print("⚠️  fpdf not installed. Run: pip install fpdf")


class Code2PDF:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output_dir = Path("full_report")
        self.output_dir.mkdir(exist_ok=True)

        # Define all files to include (in order)
        self.code_files = [
            # Config files
            ("config.json", "Configuration File"),
            (".gitignore", "Git Ignore"),
            ("requirements.txt", "Requirements"),
            # Main files
            ("main.py", "Main Pipeline"),
            # Data files
            ("data/__init__.py", "Data Package Init"),
            ("data/generate_synthetic_data.py", "Synthetic Data Generator"),
            # Source files
            ("src/__init__.py", "Source Package Init"),
            ("src/config.py", "Configuration Loader"),
            ("src/config_loader.py", "Config Cache"),
            ("src/environment.py", "Traffic Environment"),
            ("src/agent.py", "Q-Learning Agent"),
            ("src/train.py", "Training Loop"),
            ("src/evaluate.py", "Evaluation"),
            ("src/ml_models.py", "ML Models Comparison"),
            ("src/visualize.py", "Visualization"),
            ("src/report.py", "Report Generator"),
        ]

        # Output files to include
        self.output_files = [
            ("logs/training_history.json", "Training Logs"),
            ("logs/evaluation_results.json", "Evaluation Results"),
            ("logs/ml_metrics.json", "ML Metrics"),
            ("config.json", "Configuration"),
        ]

        # Charts to include
        self.chart_files = [
            ("results/learning_curve.png", "Learning Curve"),
            ("results/step_by_step_learning.png", "Step-by-Step Learning"),
            ("results/evaluation_comparison.png", "Evaluation Comparison"),
            ("results/ml_model_comparison.png", "ML Model Comparison"),
        ]

    def collect_file_content(self, filepath):
        """Read file content with error handling."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def format_code_for_pdf(self, code, language="python"):
        """Format code for PDF display."""
        lines = code.split("\n")
        formatted = []
        for i, line in enumerate(lines, 1):
            # Remove any problematic characters
            clean_line = line.encode("ascii", "replace").decode("ascii")
            formatted.append(f"{i:4d} | {clean_line}")
        return "\n".join(formatted)

    def clean_text(self, text):
        """Remove or replace problematic Unicode characters."""
        # Replace common emojis with text
        replacements = {
            "✅": "[OK]",
            "✓": "[OK]",
            "⚠": "[WARN]",
            "❌": "[ERROR]",
            "🎉": "[SUCCESS]",
            "📊": "[CHART]",
            "📄": "[FILE]",
            "📈": "[GRAPH]",
            "🤖": "[AI]",
            "🦸": "[HERO]",
            "🏆": "[WINNER]",
            "🔧": "[TOOL]",
            "🧹": "[CLEAN]",
            "→": "->",
            "•": "*",
            "–": "-",
            "—": "-",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }
        for emoji, replacement in replacements.items():
            text = text.replace(emoji, replacement)

        # Encode to ascii, replacing any remaining non-ascii chars
        return text.encode("ascii", "replace").decode("ascii")

    def generate_text_report(self):
        """Generate comprehensive text report of all outputs."""
        report = []
        report.append("=" * 80)
        report.append("TRAFFIC SIGNAL RL PROJECT - COMPLETE REPORT")
        report.append(f"Generated: {self.timestamp}")
        report.append("=" * 80)

        # Add evaluation results
        eval_path = Path("logs/evaluation_results.json")
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

            report.append("\n" + "=" * 60)
            report.append("EVALUATION RESULTS")
            report.append("=" * 60)

            for agent, metrics in eval_results.items():
                report.append(f"\n{agent}:")
                report.append(f"  Mean Reward: {metrics['mean_reward']:.3f}")
                report.append(f"  Std Reward: {metrics['std_reward']:.3f}")
                report.append(f"  Mean Queue: {metrics['mean_queue']:.2f} cars")
                report.append(f"  Std Queue: {metrics['std_queue']:.2f} cars")

        # Add ML results
        ml_path = Path("logs/ml_metrics.json")
        if ml_path.exists():
            with open(ml_path, "r", encoding="utf-8") as f:
                ml_results = json.load(f)

            report.append("\n" + "=" * 60)
            report.append("ML MODELS COMPARISON")
            report.append("=" * 60)

            for model, metrics in ml_results.items():
                report.append(f"\n{model}:")
                report.append(f"  MAE: {metrics['mae']:.3f}")
                report.append(f"  R²: {metrics['r2']:.3f}")

        # Add training summary
        train_path = Path("logs/training_history.json")
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                train_history = json.load(f)

            rewards = train_history["episode_rewards"]
            queues = train_history["mean_queue_per_ep"]

            report.append("\n" + "=" * 60)
            report.append("TRAINING SUMMARY")
            report.append("=" * 60)
            report.append(f"\nTotal Episodes: {len(rewards)}")
            report.append(f"Final Avg Reward (last 50): {np.mean(rewards[-50:]):.3f}")
            report.append(
                f"Final Avg Queue (last 50): {np.mean(queues[-50:]):.2f} cars"
            )
            report.append(f"Best Reward: {max(rewards):.3f}")
            report.append(f"Best Queue: {min(queues):.2f} cars")

        # Calculate improvement
        if (
            eval_path.exists()
            and "RL Agent" in eval_results
            and "Fixed Timer" in eval_results
        ):
            rl_queue = eval_results["RL Agent"]["mean_queue"]
            fixed_queue = eval_results["Fixed Timer"]["mean_queue"]
            improvement = (fixed_queue - rl_queue) / fixed_queue * 100

            report.append("\n" + "=" * 60)
            report.append("KEY RESULT")
            report.append("=" * 60)
            report.append(
                f"\n[OK] RL Agent improves by {improvement:.1f}% over Fixed Timer!"
            )
            report.append(f"   RL Queue: {rl_queue:.2f} cars")
            report.append(f"   Fixed Queue: {fixed_queue:.2f} cars")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def create_pdf_fpdf(self):
        """Create PDF using FPDF."""
        if not FPDF_AVAILABLE:
            print("❌ FPDF not available. Install with: pip install fpdf")
            return None

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title Page
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 40, "Traffic Signal RL Project", 0, 1, "C")
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Complete Code and Results", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated: {self.timestamp}", 0, 1, "C")

        # Table of Contents
        pdf.add_page()
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, "Table of Contents", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        toc_items = [
            "1. Configuration Files",
            "2. Main Scripts",
            "3. Source Code",
            "4. Training Results",
            "5. Evaluation Results",
            "6. ML Model Comparison",
            "7. Visualizations",
        ]
        for item in toc_items:
            pdf.cell(0, 8, item, 0, 1, "L")

        # Configuration Files
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "1. Configuration Files", 0, 1, "L")

        for filepath, description in self.code_files:
            if not Path(filepath).exists():
                continue

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"\n{description} ({filepath})", 0, 1, "L")
            pdf.set_font("Courier", "", 8)

            content = self.collect_file_content(filepath)
            # Clean content
            content = self.clean_text(content)

            # Split into lines and add to PDF
            for line in content.split("\n")[:100]:  # Limit to 100 lines per file
                try:
                    pdf.cell(0, 4, line[:95], 0, 1, "L")
                except:
                    pass

        # Results Summary
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Results Summary", 0, 1, "L")
        pdf.set_font("Courier", "", 10)

        summary = self.generate_text_report()
        summary = self.clean_text(summary)

        for line in summary.split("\n"):
            try:
                pdf.cell(0, 5, line[:95], 0, 1, "L")
            except:
                pass

        output_path = (
            self.output_dir
            / f"complete_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        pdf.output(str(output_path))
        return output_path

    def create_pdf_matplotlib(self):
        """Create PDF using matplotlib (better for including images)."""
        output_path = (
            self.output_dir
            / f"complete_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        with PdfPages(output_path) as pdf:
            # Title Page
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(
                "Traffic Signal RL Project\nComplete Code and Results",
                fontsize=24,
                y=0.7,
            )
            plt.figtext(
                0.5, 0.5, f"Generated: {self.timestamp}", ha="center", fontsize=12
            )
            plt.figtext(
                0.5,
                0.45,
                "Reinforcement Learning for Traffic Optimization",
                ha="center",
                fontsize=14,
                style="italic",
            )
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()

            # Code Files
            for filepath, description in self.code_files:
                if not Path(filepath).exists():
                    continue

                fig = plt.figure(figsize=(8.5, 11))
                plt.suptitle(f"{description}\n{filepath}", fontsize=14, y=0.95)

                content = self.collect_file_content(filepath)
                content = self.clean_text(content)
                lines = content.split("\n")

                # Create text display
                text_display = []
                for i, line in enumerate(lines[:150]):  # Limit lines
                    clean_line = line.replace(
                        "_", "\\_"
                    )  # Escape underscores for matplotlib
                    text_display.append(f"{i+1:4d} | {clean_line}")

                plt.figtext(
                    0.05,
                    0.9,
                    "\n".join(text_display),
                    fontfamily="monospace",
                    fontsize=8,
                    verticalalignment="top",
                )
                plt.axis("off")
                pdf.savefig(fig)
                plt.close()

            # Charts
            for chart_path, title in self.chart_files:
                if Path(chart_path).exists():
                    fig = plt.figure(figsize=(8.5, 11))
                    plt.suptitle(title, fontsize=16, y=0.95)

                    img = plt.imread(chart_path)
                    plt.imshow(img)
                    plt.axis("off")
                    pdf.savefig(fig)
                    plt.close()

            # Results Summary
            fig = plt.figure(figsize=(8.5, 11))
            plt.suptitle("Results Summary", fontsize=16, y=0.95)

            summary = self.generate_text_report()
            summary = self.clean_text(summary)

            plt.figtext(
                0.05,
                0.9,
                summary,
                fontfamily="monospace",
                fontsize=8,
                verticalalignment="top",
            )
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()

        return output_path

    def generate_markdown(self):
        """Generate markdown report (good for GitHub)."""
        md_lines = []
        md_lines.append("# Traffic Signal RL Project - Complete Report")
        md_lines.append(f"**Generated:** {self.timestamp}")
        md_lines.append("")

        # Table of Contents
        md_lines.append("## Table of Contents")
        md_lines.append("1. [Configuration Files](#configuration-files)")
        md_lines.append("2. [Main Scripts](#main-scripts)")
        md_lines.append("3. [Source Code](#source-code)")
        md_lines.append("4. [Results](#results)")
        md_lines.append("5. [Visualizations](#visualizations)")
        md_lines.append("")

        # Configuration Files
        md_lines.append("## Configuration Files")
        for filepath, description in self.code_files:
            if (
                "config" in filepath
                or filepath.endswith(".json")
                or filepath.endswith(".txt")
            ):
                md_lines.append(f"\n### {description} (`{filepath}`)")
                md_lines.append("```" + ("json" if filepath.endswith(".json") else ""))
                content = self.collect_file_content(filepath)
                content = self.clean_text(content)
                md_lines.append(content)
                md_lines.append("```")

        # Source Code
        md_lines.append("## Source Code")
        for filepath, description in self.code_files:
            if filepath.startswith("src/") and filepath != "src/__init__.py":
                md_lines.append(f"\n### {description} (`{filepath}`)")
                md_lines.append("```python")
                content = self.collect_file_content(filepath)
                content = self.clean_text(content)
                md_lines.append(content)
                md_lines.append("```")

        # Results
        md_lines.append("## Results")

        # Evaluation Results
        eval_path = Path("logs/evaluation_results.json")
        if eval_path.exists():
            md_lines.append("\n### Evaluation Results")
            md_lines.append("```")
            with open(eval_path, "r", encoding="utf-8") as f:
                content = f.read()
                content = self.clean_text(content)
                md_lines.append(content)
            md_lines.append("```")

        # Training Summary
        train_path = Path("logs/training_history.json")
        if train_path.exists():
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = json.load(f)

            md_lines.append("\n### Training Summary")
            md_lines.append(f"- **Episodes:** {len(train_data['episode_rewards'])}")
            md_lines.append(
                f"- **Final Avg Reward:** {np.mean(train_data['episode_rewards'][-50:]):.3f}"
            )
            md_lines.append(
                f"- **Final Avg Queue:** {np.mean(train_data['mean_queue_per_ep'][-50:]):.2f} cars"
            )

        # Visualizations
        md_lines.append("\n## Visualizations")
        for chart_path, title in self.chart_files:
            if Path(chart_path).exists():
                md_lines.append(f"\n### {title}")
                md_lines.append(f"![{title}]({chart_path})")

        # Save markdown
        md_path = (
            self.output_dir
            / f"complete_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return md_path

    def generate_all(self):
        """Generate all report formats."""
        print("\n" + "=" * 60)
        print("📄 GENERATING COMPLETE PROJECT REPORT")
        print("=" * 60)

        # Generate text summary
        summary = self.generate_text_report()
        summary = self.clean_text(summary)
        summary_path = (
            self.output_dir
            / f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[OK] Text summary: {summary_path}")

        # Generate markdown
        md_path = self.generate_markdown()
        print(f"[OK] Markdown report: {md_path}")

        # Generate PDF (try both methods)
        pdf_path = None
        if FPDF_AVAILABLE:
            try:
                pdf_path = self.create_pdf_fpdf()
                if pdf_path:
                    print(f"[OK] PDF report (FPDF): {pdf_path}")
            except Exception as e:
                print(f"[WARN] FPDF generation failed: {e}")

        try:
            pdf_path2 = self.create_pdf_matplotlib()
            print(f"[OK] PDF report (Matplotlib): {pdf_path2}")
        except Exception as e:
            print(f"[WARN] Matplotlib PDF generation failed: {e}")

        print("\n" + "=" * 60)
        print(f"[OK] All reports saved to: {self.output_dir}/")
        print("=" * 60)

        return {"summary": summary_path, "markdown": md_path, "pdf": pdf_path}


def main():
    """Main function to generate complete report."""
    generator = Code2PDF()
    results = generator.generate_all()

    # Print preview of summary
    print("\n📋 Report Preview:")
    print("-" * 40)
    with open(results["summary"], "r", encoding="utf-8") as f:
        preview = f.readlines()[:20]
        for line in preview:
            print(line.rstrip())
    print("-" * 40)
    print(f"\nFull report available at: {results['summary']}")


if __name__ == "__main__":
    main()
