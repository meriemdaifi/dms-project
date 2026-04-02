"""
DMS Main Entry Point
=====================
Runs the complete DMS simulation pipeline with all scenarios
and generates validation results and plots.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

from src.ecu_decision import ECUSimulator
from src.scenarios import (
    generate_attentive_scenario,
    generate_fatigued_scenario,
    generate_distracted_scenario,
    generate_mixed_scenario,
)
from src.visualization import (
    create_scenario_dataframe,
    plot_fatigue_analysis,
    plot_distraction_analysis,
    plot_combined_dashboard,
    generate_summary_table,
)


def run_all_scenarios(output_dir: str = "results") -> None:
    """
    Run all test scenarios and generate results.

    Args:
        output_dir: Directory to save results and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    simulator = ECUSimulator()
    fps = 30.0
    time_step = 1.0 / fps

    scenarios = {
        "attentive": {
            "generator": generate_attentive_scenario,
            "duration": 120.0,
            "description": "Normal attentive driving",
        },
        "fatigued": {
            "generator": generate_fatigued_scenario,
            "duration": 120.0,
            "description": "Progressive fatigue scenario",
        },
        "distracted": {
            "generator": generate_distracted_scenario,
            "duration": 120.0,
            "description": "Increasing distraction scenario",
        },
        "mixed": {
            "generator": generate_mixed_scenario,
            "duration": 180.0,
            "description": "Mixed scenario with state transitions",
        },
    }

    all_summaries = []

    for name, config in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Running scenario: {name}")
        print(f"Description: {config['description']}")
        print(f"Duration: {config['duration']}s at {fps} FPS")
        print(f"{'='*60}")

        start_time = time.time()

        # Generate scenario data
        scenario_data = config["generator"](
            duration_seconds=config["duration"], fps=fps
        )

        # Run simulation
        results = simulator.run_scenario(scenario_data, time_step=time_step)

        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f}s")

        # Convert to DataFrame
        df = create_scenario_dataframe(results)

        # Save CSV
        csv_path = os.path.join(output_dir, f"{name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Generate plots
        plot_fatigue_analysis(
            df,
            title=f"Fatigue Analysis - {name.title()} Scenario",
            save_path=os.path.join(output_dir, "plots", f"{name}_fatigue.png"),
        )
        plot_distraction_analysis(
            df,
            title=f"Distraction Analysis - {name.title()} Scenario",
            save_path=os.path.join(output_dir, "plots", f"{name}_distraction.png"),
        )
        plot_combined_dashboard(
            df,
            title=f"DMS Dashboard - {name.title()} Scenario",
            save_path=os.path.join(output_dir, "plots", f"{name}_dashboard.png"),
        )

        # Generate summary
        summary = generate_summary_table(df)
        summary["Scenario"] = name
        all_summaries.append(summary)

        # Print key metrics
        print(f"\nKey Metrics:")
        print(f"  Mean Fatigue Score: {df['fatigue_score'].mean():.3f}")
        print(f"  Max Fatigue Score:  {df['fatigue_score'].max():.3f}")
        print(f"  Mean Distraction:   {df['distraction_score'].mean():.3f}")
        print(f"  Max Distraction:    {df['distraction_score'].max():.3f}")
        print(f"  Mean PERCLOS:       {df['perclos'].mean():.1f}%")
        print(f"  Total Alerts:       {df['total_alerts'].max()}")

        # Validate expected behavior
        validate_scenario(name, df)

    # Save combined summary
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        summary_path = os.path.join(output_dir, "combined_summary.csv")
        combined.to_csv(summary_path, index=False)
        print(f"\nCombined summary saved to {summary_path}")

    print(f"\n{'='*60}")
    print("All scenarios completed successfully!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")


def validate_scenario(name: str, df: pd.DataFrame) -> None:
    """
    Validate that scenario results match expected behavior.

    Args:
        name: Scenario name.
        df: Results DataFrame.
    """
    print(f"\nValidation for '{name}' scenario:")

    if name == "attentive":
        mean_fatigue = df["fatigue_score"].mean()
        mean_distraction = df["distraction_score"].mean()
        attentive_pct = (df["driver_state"] == "attentive").mean() * 100

        checks = [
            ("Low fatigue score (<0.3)", mean_fatigue < 0.3),
            ("Low distraction score (<0.3)", mean_distraction < 0.3),
            ("High attentive percentage (>70%)", attentive_pct > 70),
        ]

    elif name == "fatigued":
        max_fatigue = df["fatigue_score"].max()
        # Check later portion of scenario
        late_df = df[df["time"] > df["time"].max() * 0.7]
        late_fatigue = late_df["fatigue_score"].mean()

        checks = [
            ("Fatigue detected (max >0.3)", max_fatigue > 0.3),
            ("Late-stage high fatigue (>0.2)", late_fatigue > 0.2),
            ("PERCLOS increases", df["perclos"].iloc[-100:].mean() > df["perclos"].iloc[:100].mean()),
        ]

    elif name == "distracted":
        max_distraction = df["distraction_score"].max()
        distracted_pct = (df["driver_state"] == "distracted").mean() * 100

        checks = [
            ("Distraction detected (max >0.3)", max_distraction > 0.3),
            ("Distracted state observed", distracted_pct > 0),
        ]

    elif name == "mixed":
        checks = [
            ("Multiple states observed", df["driver_state"].nunique() >= 2),
            ("Total alerts > 0", df["total_alerts"].max() > 0),
        ]

    else:
        checks = []

    for description, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {description}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    run_all_scenarios(output_dir)
