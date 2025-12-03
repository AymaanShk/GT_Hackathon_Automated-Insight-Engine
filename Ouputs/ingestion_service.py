# ingestion_service.py

import time
import polars as pl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

# 💡 NEW IMPORT: Import the function from your separate logic file
from etl_logic import transform_and_analyze 

# --- Configuration ---
INPUT_DIR = "./input_files"

# ⚠️ Define the STRICT SCHEMA for production-readiness
# This ensures consistency and prevents bugs from bad input files.
EXPECTED_SCHEMA = {
    "Date": pl.Date,
    "Campaign_ID": pl.Int64,
    "Spend": pl.Float64,
    "Impressions": pl.Int64,
    "Conversions": pl.Int64,
    "Location": pl.Utf8 # Location is critical for anomaly drill-down
}

# --- Core Processing Function (Will trigger Phase 2) ---

def process_data(file_path: str):
    """Reads the CSV using Polars and starts the ETL pipeline."""
    try:
        # Use Polars to read with the strict, defined schema
        df = pl.read_csv(file_path, dtypes=EXPECTED_SCHEMA)
        
        print(f"\n✅ Data ingested successfully from: {file_path}")
        print(f"   Shape: {df.shape}")
        
        # 💡 UPDATED LOGIC: Call the next phase (Transformation/Anomaly)
        print("--- Starting Phase 2: Transformation and Anomaly Detection ---")
        df_weekly, anomalies = transform_and_analyze(df, file_path)
        
        # Optional: Print results for confirmation
        # print("\nWoW Metrics (Sample):\n", df_weekly.tail(5))
        # print("\nDetected Anomalies:\n", anomalies)
        
    except pl.ComputeError as e:
        print(f"\n❌ ERROR: Schema validation failed for {file_path}. Check data types.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ A general error occurred during ingestion: {e}")
        
    # Optional: Move the processed file to an 'archive' folder after processing

# --- Watchdog Implementation ---
# ... (CSVHandler and start_watcher functions remain the same as you provided) ...

class CSVHandler(FileSystemEventHandler):
    """Handles file system events for the input directory."""
    def on_created(self, event):
        """Called when a file or directory is created."""
        if not event.is_directory and event.src_path.lower().endswith('.csv'):
            # Wait a moment to ensure the file is fully written/uploaded before reading
            time.sleep(1) 
            process_data(event.src_path)

def start_watcher():
    """Sets up and starts the Watchdog observer."""
    print(f"--- Starting Watchdog Service ---")
    print(f"Monitoring directory: {INPUT_DIR}")
    
    event_handler = CSVHandler()
    observer = Observer()
    
    # Schedule the handler to watch the INPUT_DIR
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()