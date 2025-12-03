🚀 TrendSpotter: The Automated Insight Engine
Tagline: An event-driven data pipeline that converts raw CSV logs into executive-ready PDF reports with AI-generated narratives in under 30 seconds.

1. The Problem (Real World Scenario)
Context: During my research into AdTech workflows, I identified a major inefficiency: Account Managers waste 4-6 hours every week manually downloading CSVs and taking screenshots to create "Weekly Performance Reports."

The Pain Point: This manual process is slow, boring, and error-prone. If a campaign is wasting budget, the client might not know for days because the reporting lag is too high.

My Solution: I built TrendSpotter, an event-driven system. You simply drop a raw file into a folder, and 30 seconds later, you receive a fully analyzed, executive-ready PDF report in your email.

2. Expected End Result
For the User:

Input: Drop a raw CSV file into the folder.

Action: Wait 30 seconds.

Output: Receive a professionally formatted PDF via email containing:

Week-over-Week growth charts.

A list of detected anomalies (e.g., "Traffic dropped 40% in Miami").

An AI-written paragraph explaining why the drop happened (correlated with Weather API).

3. Technical Approach
I wanted to challenge myself to build a system that is Production-Ready, moving beyond simple scripts to a robust ETL (Extract, Transform, Load) pipeline.

System Architecture:

Ingestion (Event-Driven): A Python watchdog script listens for file changes in real-time.

Decision: I chose Polars over Pandas because it handles larger datasets faster and enforces a stricter schema, which reduces bugs in production.

Anomaly Detection: I implemented the Isolation Forest algorithm (Scikit-Learn) to mathematically identify "weird" data points (outliers) rather than just using hard-coded if/else rules.

Generative AI (The Analyst):

We pass the anomaly metadata to Google Gemini 1.5 Pro.

We use a Few-Shot Prompt technique to force the AI to sound like a Senior Data Analyst.

Guardrail: I implemented a validation step to ensure the AI's math matches the Polars dataframe to prevent hallucinations.

Reporting: WeasyPrint renders the final HTML/CSS report into a pixel-perfect PDF.

4. Tech Stack
Language: Python 3.11

Data Engine: Polars (Rust-based DataFrame library)

Machine Learning: Scikit-Learn (Isolation Forest)

AI Model: Google Gemini 1.5 Pro (via Vertex AI)

Orchestration: Docker & Docker Compose

Visualization: Plotly & WeasyPrint

5. Challenges & Learnings
This project wasn't easy. Here are two major hurdles I overcame:

Challenge 1: AI Hallucinations

Issue: Initially, the AI would invent reasons for data drops (e.g., claiming "It rained" when I provided no weather data).

Solution: I implemented a "Strict Context" System Prompt. I effectively told the AI: "Only use the data provided in the JSON context. If you don't know, say 'Unknown'." This reduced hallucination rates significantly.

Challenge 2: Docker Networking

Issue: My Python container couldn't send emails out to the SMTP server.

Solution: I learned about Docker Networks and container isolation. I had to configure the SMTP ports and environment variables correctly in the docker-compose.yml to allow external traffic.

6. Visual Proof
Anomaly Detected (Terminal)

Final Report (PDF)

Terminal showing Isolation Forest detecting outliers

Final Output sent to client via Email

7. How to Run
Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML  # 1. Clone Repository  git clone [(https://github.com/AymaanShk/GT_Hackathon_Automated-Insight-Engine)](https://github.com/username/trendspotter.git)  # 2. Add API Key  export GEMINI_API_KEY="your_key_here"  # 3. Build & Run Container  docker-compose up --build  # 4. Test  # Move the sample.csv to the input folder to trigger the pipeline  mv data/sample.csv data/input/  
