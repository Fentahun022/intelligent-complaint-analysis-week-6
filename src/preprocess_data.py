# src/preprocess_data.py
import os
import sys

# --- Path Modification ---
# This must be at the top of the file, before any `from src...` imports.
# This line adds the project's root directory (the parent of 'src') to Python's path.
# This makes the import 'from src.config...' work, regardless of where the script is run from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Now, we can safely import from our project ---
import polars as pl
from src.config import RAW_DATA_PATH, FILTERED_DATA_PATH

def preprocess_with_batches(sample_size=5000, batch_size=50000):
    """
    Reads the raw CSV in batches, processes each batch in memory,
    and combines the results for a memory-efficient workflow.
    """
    if os.path.exists(FILTERED_DATA_PATH):
        print(f"Cleaned data already exists at {FILTERED_DATA_PATH}. Skipping.")
        return

    print(f"--- Starting BATCH-BASED Data Preprocessing (Batch Size: {batch_size}) ---")

    processed_batches = []
    reader = pl.read_csv_batched(RAW_DATA_PATH, batch_size=batch_size, ignore_errors=True)

    print("Processing batches...")
    batch_num = 1
    try:
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            
            batch_df = batches[0]
            print(f"  - Processing Batch {batch_num} ({len(batch_df)} rows)...")

            target_products = [
                'Credit card or prepaid card', 'Checking or savings account',
                'Money transfer, virtual currency, or money service',
                'Payday loan, title loan, or personal loan',
                'Credit reporting, credit repair services, or other personal consumer reports'
            ]

            processed_batch = (
                batch_df
                .filter(
                    pl.col('Product').is_in(target_products) &
                    pl.col('Consumer complaint narrative').is_not_null()
                )
                .with_columns(
                    pl.col('Consumer complaint narrative').str.to_lowercase()
                    .str.replace_all(r"x{2,}", "").str.replace_all(r"[^a-z0-9\s.,]", "")
                    .str.replace_all(r"\s+", " ").alias("cleaned_narrative")
                )
                .select(['Complaint ID', 'Product', 'cleaned_narrative'])
            )

            if not processed_batch.is_empty():
                processed_batches.append(processed_batch)
            
            batch_num += 1
    except StopIteration:
        pass # Expected end of iteration

    print("All batches processed.")

    if not processed_batches:
        print("Warning: No data matched the filtering criteria. No output file will be created.")
        return

    final_filtered_df = pl.concat(processed_batches)
    print(f"Total filtered records: {len(final_filtered_df)}")

    if len(final_filtered_df) > sample_size:
        final_df = final_filtered_df.sample(n=sample_size, seed=42, shuffle=True)
        print(f"Sampled {len(final_df)} records for the final dataset.")
    else:
        final_df = final_filtered_df

    final_df.write_csv(FILTERED_DATA_PATH)
    print(f"✅ Preprocessing complete. Cleaned data saved to: {FILTERED_DATA_PATH}")

# This block ensures the script can be run directly
if __name__ == "__main__":
    # Check if the raw data file exists before attempting to run
    # The config variables are already imported at the top of the file
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data file not found at '{RAW_DATA_PATH}'.")
        print("Please ensure 'complaints.csv' is in the 'data' directory.")
    else:
        preprocess_with_batches()