from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path('..').resolve() / 'dp_arve'
file_path = BASE_DIR / 'outputs/OFEV_probabilistic/station2170_deterministic.csv'
output_path = BASE_DIR / "outputs" / "ofev_median_of_medians.csv"

# Load data
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Rename for clarity
df = df.rename(columns={
    'issue_time': 'forecast_time',
    'lead_time_h': 'lead_time'
})

# Convert types
df['Q'] = pd.to_numeric(df['Q'], errors='coerce')
df['valid_time'] = pd.to_datetime(df['valid_time'])
df['forecast_time'] = pd.to_datetime(df['forecast_time'])
df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce')

# Drop missing
df = df.dropna(subset=['Q', 'valid_time', 'forecast_time', 'lead_time'])

# Median across models
median_df = (
    df.groupby(['forecast_time', 'valid_time', 'lead_time'], as_index=False)
      .agg(
          Q_ofev=('Q', 'median'),
          n_models=('Q', 'count')
      )
)

# Sort
median_df = median_df.sort_values(
    ['forecast_time', 'valid_time', 'lead_time']
).reset_index(drop=True)

# Save
output_path.parent.mkdir(parents=True, exist_ok=True)
median_df.to_csv(output_path, index=False)

print(median_df.head())
print(f"Saved to {output_path}")