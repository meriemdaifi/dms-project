"""
Visualization and Results Module
==================================
Generates plots and reports for the DMS validation results
using Matplotlib and Pandas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional
from pathlib import Path


def create_scenario_dataframe(results: List[dict]) -> pd.DataFrame:
    """Convert simulation results to a Pandas DataFrame."""
    return pd.DataFrame(results)


def plot_fatigue_analysis(
    df: pd.DataFrame,
    title: str = "Fatigue Analysis",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot fatigue-related metrics over time.

    Creates a multi-panel figure showing:
    - Eye state (open/closed) over time
    - PERCLOS percentage
    - Fatigue score with threshold lines
    - Alert level

    Args:
        df: DataFrame with simulation results.
        title: Plot title.
        save_path: Path to save the figure (optional).
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    time_axis = df["time"]

    # Panel 1: Eye state
    axes[0].fill_between(
        time_axis,
        df["eyes_open"].astype(int),
        alpha=0.5,
        color="green",
        label="Eyes Open",
    )
    axes[0].set_ylabel("Eye State")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Closed", "Open"])
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: PERCLOS
    axes[1].plot(time_axis, df["perclos"], color="orange", linewidth=1.5)
    axes[1].axhline(y=40, color="red", linestyle="--", alpha=0.7, label="Fatigue Threshold (40%)")
    axes[1].axhline(y=70, color="darkred", linestyle="--", alpha=0.7, label="Drowsy Threshold (70%)")
    axes[1].set_ylabel("PERCLOS (%)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Fatigue score
    axes[2].plot(time_axis, df["fatigue_score"], color="red", linewidth=1.5, label="Fatigue Score")
    axes[2].axhline(y=0.35, color="orange", linestyle="--", alpha=0.7, label="Warning (0.35)")
    axes[2].axhline(y=0.6, color="red", linestyle="--", alpha=0.7, label="Critical (0.6)")
    axes[2].axhline(y=0.75, color="darkred", linestyle="--", alpha=0.7, label="Emergency (0.75)")
    axes[2].set_ylabel("Fatigue Score")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Alert level
    colors_map = {0: "green", 1: "blue", 2: "orange", 3: "red", 4: "darkred"}
    colors = [colors_map.get(int(level), "gray") for level in df["alert_level"]]
    axes[3].scatter(time_axis, df["alert_level"], c=colors, s=10, alpha=0.7)
    axes[3].set_ylabel("Alert Level")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_yticks([0, 1, 2, 3, 4])
    axes[3].set_yticklabels(["None", "Info", "Warning", "Critical", "Emergency"])
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distraction_analysis(
    df: pd.DataFrame,
    title: str = "Distraction Analysis",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distraction-related metrics over time.

    Args:
        df: DataFrame with simulation results.
        title: Plot title.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    time_axis = df["time"]

    # Panel 1: Head yaw
    axes[0].plot(time_axis, df["yaw"], color="blue", linewidth=1.0, label="Yaw")
    axes[0].axhline(y=25, color="red", linestyle="--", alpha=0.5)
    axes[0].axhline(y=-25, color="red", linestyle="--", alpha=0.5)
    axes[0].fill_between(time_axis, -25, 25, alpha=0.1, color="green", label="Safe Zone")
    axes[0].set_ylabel("Yaw (degrees)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Distraction score
    axes[1].plot(
        time_axis, df["distraction_score"], color="purple", linewidth=1.5, label="Distraction Score"
    )
    axes[1].axhline(y=0.35, color="orange", linestyle="--", alpha=0.7, label="Warning (0.35)")
    axes[1].axhline(y=0.6, color="red", linestyle="--", alpha=0.7, label="Critical (0.6)")
    axes[1].set_ylabel("Distraction Score")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Driver state
    state_map = {"attentive": 0, "fatigued": 1, "distracted": 2, "drowsy": 3, "unknown": -1}
    state_numeric = [state_map.get(s, -1) for s in df["driver_state"]]
    state_colors = {
        0: "green",
        1: "orange",
        2: "purple",
        3: "red",
        -1: "gray",
    }
    colors = [state_colors.get(s, "gray") for s in state_numeric]
    axes[2].scatter(time_axis, state_numeric, c=colors, s=10, alpha=0.7)
    axes[2].set_ylabel("Driver State")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(["Attentive", "Fatigued", "Distracted", "Drowsy"])
    axes[2].grid(True, alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color="green", label="Attentive"),
        mpatches.Patch(color="orange", label="Fatigued"),
        mpatches.Patch(color="purple", label="Distracted"),
        mpatches.Patch(color="red", label="Drowsy"),
    ]
    axes[2].legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined_dashboard(
    df: pd.DataFrame,
    title: str = "DMS Dashboard",
    save_path: Optional[str] = None,
) -> None:
    """
    Create a combined dashboard showing all key metrics.

    Args:
        df: DataFrame with simulation results.
        title: Plot title.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    time_axis = df["time"]

    # Top-left: Fatigue and Distraction scores
    axes[0, 0].plot(time_axis, df["fatigue_score"], "r-", linewidth=1.5, label="Fatigue")
    axes[0, 0].plot(time_axis, df["distraction_score"], "b-", linewidth=1.5, label="Distraction")
    axes[0, 0].set_title("Fatigue vs Distraction Scores")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.05, 1.05)

    # Top-right: PERCLOS
    axes[0, 1].plot(time_axis, df["perclos"], "orange", linewidth=1.5)
    axes[0, 1].axhline(y=40, color="red", linestyle="--", alpha=0.5, label="Threshold")
    axes[0, 1].set_title("PERCLOS Over Time")
    axes[0, 1].set_ylabel("PERCLOS (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Eye state
    axes[1, 0].fill_between(time_axis, df["eyes_open"].astype(int), alpha=0.5, color="green")
    axes[1, 0].set_title("Eye State")
    axes[1, 0].set_ylabel("State")
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(["Closed", "Open"])
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Head yaw
    axes[1, 1].plot(time_axis, df["yaw"], "b-", linewidth=1.0)
    axes[1, 1].axhline(y=25, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].axhline(y=-25, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Head Yaw Angle")
    axes[1, 1].set_ylabel("Yaw (degrees)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary statistics table from simulation results.

    Args:
        df: DataFrame with simulation results.

    Returns:
        Summary DataFrame.
    """
    summary = {
        "Metric": [
            "Total Duration (s)",
            "Mean Fatigue Score",
            "Max Fatigue Score",
            "Mean Distraction Score",
            "Max Distraction Score",
            "Mean PERCLOS (%)",
            "Max PERCLOS (%)",
            "% Time Eyes Closed",
            "% Time Attentive",
            "% Time Fatigued",
            "% Time Distracted",
            "% Time Drowsy",
            "Total Alerts Generated",
        ],
        "Value": [
            f"{df['time'].max():.1f}",
            f"{df['fatigue_score'].mean():.3f}",
            f"{df['fatigue_score'].max():.3f}",
            f"{df['distraction_score'].mean():.3f}",
            f"{df['distraction_score'].max():.3f}",
            f"{df['perclos'].mean():.1f}",
            f"{df['perclos'].max():.1f}",
            f"{(1 - df['eyes_open'].mean()) * 100:.1f}",
            f"{(df['driver_state'] == 'attentive').mean() * 100:.1f}",
            f"{(df['driver_state'] == 'fatigued').mean() * 100:.1f}",
            f"{(df['driver_state'] == 'distracted').mean() * 100:.1f}",
            f"{(df['driver_state'] == 'drowsy').mean() * 100:.1f}",
            f"{df['total_alerts'].max()}",
        ],
    }
    return pd.DataFrame(summary)
