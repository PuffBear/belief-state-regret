import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_plots_from_excel(excel_path='checkpoints/training_log.xlsx', output_path='checkpoints/training_curves_from_excel.png'):
    print(f"Loading metrics from {excel_path}")
    
    # Read the Excel file
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Sanitize dataframe to drop string summaries (e.g. "full:")
    df['Episode'] = pd.to_numeric(df['Episode'], errors='coerce')
    df = df.dropna(subset=['Episode'])
    
    if df.empty:
        print("No valid episodes found in the Excel file.")
        return

    # Use seaborn for styling
    sns.set_theme(style="darkgrid")
    
    # The Excel file has 4 primary metrics we can plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('B-SRM-CHFA Training Metrics', fontsize=18, fontweight='bold', y=0.98)

    # 1. Capture Rate
    if 'Capture Rate (%)' in df.columns:
        axes[0, 0].plot(df['Episode'], df['Capture Rate (%)'], color='#1f77b4', linewidth=2.5)
        axes[0, 0].set_ylabel('Capture Rate (%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Evader Capture Rate over Training', fontsize=14)
        axes[0, 0].set_ylim(0, 100)
    
    # 2. Avg Capture Time
    if 'Avg Capture Time' in df.columns:
        axes[0, 1].plot(df['Episode'], df['Avg Capture Time'], color='#ff7f0e', linewidth=2.5)
        axes[0, 1].set_ylabel('Steps', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Average Capture Time', fontsize=14)

    # 3. Avg Cost
    if 'Avg Cost' in df.columns:
        axes[1, 0].plot(df['Episode'], df['Avg Cost'], color='#2ca02c', linewidth=2.5)
        axes[1, 0].set_ylabel('Total Cost', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Average Episode Cost', fontsize=14)
        axes[1, 0].set_xlabel('Training Episode', fontsize=12)

    # 4. Timeout Rate
    if 'Timeout Rate (%)' in df.columns:
        axes[1, 1].plot(df['Episode'], df['Timeout Rate (%)'], color='#d62728', linewidth=2.5)
        axes[1, 1].set_ylabel('Timeout Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Timeout Rate (Max Steps Reached)', fontsize=14)
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_xlabel('Training Episode', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    
    print(f"\n==========================================")
    print(f"Success! High-resolution training curves reconstructed from Excel.")
    print(f" -> {output_path}")
    print(f"==========================================")

if __name__ == '__main__':
    generate_plots_from_excel()
