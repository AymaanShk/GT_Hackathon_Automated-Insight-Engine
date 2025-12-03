# etl_logic.py

import polars as pl
from sklearn.ensemble import IsolationForest
import numpy as np

# 💡 NEW IMPORTS for Phase 3
import os
from google import genai
from weasyprint import HTML
from datetime import datetime
# Note: 'requests' is usually needed for real Weather API, but we'll simulate for now.


# --- 1. Core Transformation and WoW Metrics ---

def calculate_weekly_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates weekly spend and Week-over-Week change for reporting."""
    
    # 1. Group data by week and sum the key metric (Spend)
    df_weekly = df.group_by_dynamic("Date", every="1w").agg(
        pl.col("Spend").sum().alias("Weekly_Spend"),
    ).sort("Date")
    
    # 2. Calculate WoW percentage change
    df_weekly = df_weekly.with_columns(
        (
            ((pl.col("Weekly_Spend") / pl.col("Weekly_Spend").shift(1) - 1) * 100)
            .round(2)
            .alias("WoW_Change_%")
        )
    )
    print(f"   => Calculated Weekly WoW metrics. (First WoW value will be null/inf)")
    return df_weekly

# --- 2. Anomaly Detection using Isolation Forest ---

def detect_anomalies(df: pl.DataFrame) -> pl.DataFrame:
    """Uses Isolation Forest to detect outliers in the daily performance data."""
    
    # Isolate key numerical metrics for the model
    features_df = df.select(['Spend', 'Impressions', 'Conversions'])
    features = features_df.to_numpy()
    
    # Setup and Fit the Isolation Forest model
    # contamination=0.05 means we expect up to 5% of the data points to be outliers
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Predict: 1 for inlier, -1 for outlier
    outlier_predictions = model.fit_predict(features)
    
    # Filter the original DataFrame for anomalies (-1)
    anomaly_df = df.with_columns(
        pl.lit(outlier_predictions).alias("is_anomaly")
    ).filter(pl.col("is_anomaly") == -1)
    
    if anomaly_df.is_empty():
        print("   => No anomalies detected by Isolation Forest (Good job!).")
        # Return a structure that the AI can handle if no anomalies are found
        return pl.DataFrame({"Location": [None], "Anomaly_Count": [0], "Avg_Anomaly_Spend": [0.0]})
    
    # Summarize anomalies for the AI analyst prompt
    anomalies_summary = anomaly_df.group_by(["Location"]).agg(
        pl.count().alias("Anomaly_Count"),
        pl.col("Spend").mean().round(2).alias("Avg_Anomaly_Spend"),
    ).sort("Anomaly_Count", descending=True)
    
    print(f"   => Anomaly detection found {len(anomaly_df)} outlier(s) in {len(anomalies_summary)} location(s).")
    
    return anomalies_summary


# --- 4. The AI Analyst (Gemini 1.5 Pro) ---

def fetch_external_context(anomalies_summary: pl.DataFrame) -> str:
    """Simulates fetching external data (like Weather API) for context."""
    # Since we can't call a real weather API here, we use a simulation.
    if anomalies_summary['Anomaly_Count'][0] == 0:
        return "No anomalies detected, no external context needed."
    
    # Hardcoded simulation for the prompt based on the test data (assuming Miami is the location)
    location = anomalies_summary.get_column('Location').to_list()[0]
    
    weather_context = f"External data shows that on the day(s) of the anomaly in **{location}**, there was a severe thunderstorm warning, often correlated with low foot traffic and reduced ad conversion rates."
    
    return weather_context

def generate_analysis(anomalies_summary: pl.DataFrame) -> str:
    """Uses Gemini 1.5 Pro with a Few-Shot Prompt to write the analysis."""
    
    if not os.getenv("GEMINI_API_KEY"):
        return "AI Analysis placeholder: GEMINI_API_KEY not set. Cannot run analysis."
        
    client = genai.Client()
    
    # Convert Polars summary to a format the AI can easily read
    anomalies_data_str = anomalies_summary.write_json(row_oriented=True)
    weather_context = fetch_external_context(anomalies_summary)
    
    # Few-Shot Prompt Structure to force the persona and tone
    prompt = f"""
    SYSTEM INSTRUCTION: You are a **Senior Data Analyst**. Your response must be extremely concise, professional, and actionable. Your analysis must not exceed 100 words. Do not include any titles or introductory phrases like 'Senior Data Analyst:'.

    FEW-SHOT EXAMPLE:
    Input: Anomalies detected in Location: Dallas, Avg_Anomaly_Spend: $500. External Context: Local sports team won a championship, leading to high consumer spending but low campaign-specific conversions.
    Output: Dallas showed an unexpected spending spike (+15%) paired with suppressed conversions. This correlates with major local sporting events diverting user attention. **Actionable Insight:** Pause non-event-related campaigns immediately during major local events to conserve budget.
    
    CURRENT DATA:
    Performance Anomalies (JSON Summary): {anomalies_data_str}
    Correlated External Data (Weather/Context): {weather_context}
    
    TASK: Write a 1-paragraph explanation of the detected anomaly, correlating the performance drop (or spike) with the external context. Focus on the 'why' and suggest a clear, concise action.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"AI Analysis failed. Please check your API key and connection. Error: {e}"

# --- 5. Reporting (WeasyPrint) ---

def generate_pdf_report(df_weekly: pl.DataFrame, ai_analysis_text: str, file_path: str):
    """Renders the HTML template into a pixel-perfect PDF using WeasyPrint."""
    
    # Simple HTML structure for the report
    html_content = f"""
    <html>
    <head>
        <title>Weekly Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ color: #007bff; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h2 {{ color: #333; margin-top: 20px; }}
            .analysis {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .footer {{ margin-top: 40px; font-size: 0.8em; color: #777; }}
        </style>
    </head>
    <body>
        <h1>📊 Weekly Performance Report</h1>
        
        <h2>🤖 AI Analyst Summary (The WHY)</h2>
        <div class="analysis">
            <p>{ai_analysis_text}</p>
        </div>

        <h2>📈 Week-over-Week Spend Change</h2>
        <table>
            <thead>
                <tr><th>Date</th><th>Weekly Spend</th><th>WoW Change %</th></tr>
            </thead>
            <tbody>
            {
                "".join(
                    f"<tr><td>{row[0]}</td><td>${row[1]:.2f}</td><td>{row[2]:.2f}%</td></tr>"
                    for row in df_weekly.rows()
                )
            }
            </tbody>
        </table>
        
        <div class="footer">
            <p>Report Generated by Automated ETL Pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    output_filename = os.path.basename(file_path).replace(".csv", "_Report.pdf")
    # Save the PDF in the same directory as the input CSV
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    
    HTML(string=html_content).write_pdf(output_path)
    print(f"\n✅ PDF Report Generated and Saved to: {output_path}")

# --- 3. Main ETL Entry Point (UPDATED) ---

def transform_and_analyze(df: pl.DataFrame, file_path: str):
    """Orchestrates the transformation and analysis steps and generates the report."""
    
    df_weekly = calculate_weekly_metrics(df)
    anomalies_summary = detect_anomalies(df)
    
    print("--- Starting Phase 3: AI Analysis and PDF Reporting ---")
    
    # 1. Generate AI Analysis
    ai_analysis_text = generate_analysis(anomalies_summary)
    
    # 2. Generate the final PDF
    generate_pdf_report(df_weekly, ai_analysis_text, file_path)
    
    return df_weekly, anomalies_summary