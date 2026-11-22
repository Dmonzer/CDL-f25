"""
This is a sample script which uses the ollama package to run a large language model and generate a series of stories
based on an author's style, a genre, and a topic. This template will help you get started for your own projects.

created by efrén cruz cortés and Sophie van Genderen.
Modified by Dana Monzer for the application. 
"""

# :: IMPORTS ::
import ollama
import pandas as pd
from datetime import datetime
from pathlib import Path
import tomli as tomllib
from ollama import Client
import os
import json
import re
# ----------------- CONNECT -----------------
#client = Client(host="http://localhost:" + os.environ.get("OLLAMA_PORT", "11434")) # default port if not set)


# ----------------- LOAD CONFIG & MODEL -----------------
master_directory = Path("get_scores.toml")
with open(master_directory, "rb") as f:
    config_params = tomllib.load(f)
    print(config_params)

llm_model = config_params["model"]["llm_model"]
# pull model locally to ensure availability


import os
ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
client = ollama.Client(host=ollama_host)

#client = ollama.Client(host="http://127.0.0.1:11434")
#client.pull(llm_model)
client.pull(llm_model)

# If you have a system message in your TOML (as in your example)
sysm_file = Path(config_params["system"]["system_message"])
with open(sysm_file, "r") as f:
    sysm = f.read()

# ----------------- CATEGORIES (your 10 categories) -----------------
company_categories = [
    "Earnings & Financial Health",
    "Regulatory & Legal Risk",
    "Innovation & Product Pipeline",
    "Competitor & Market Positioning",
    "Management & Leadership",
    "Macroeconomic Headwinds/Tailwinds",
    "Supply Chain & Cost Efficiency",
    "Customer Demand & Sentiment",
    "Capital Allocation (M&A/Buybacks)",
    "ESG & Public Relations (P.R.)"
]

# ----------------- HELPERS -----------------

def build_prompt(company_name: str, news_text: str) -> str:
    return f"""
Analyze the impact of this piece of world news on the {company_name} company for each category separately. 

Output ONLY 10 integers between -5 and 5, where a negative number means that the world news has a negative impact on the company in that particular category, and a positive number means that the world news has a positive impact on the company in that particular category. 
Space-separated.
No bullet points.
No labels.
No explanations.
No text before or after.
Only 10 integers in this exact order:

1. Earnings & Financial Health
2. Regulatory & Legal Risk
3. Innovation & Product Pipeline
4. Competitor & Market Positioning
5. Management & Leadership
6. Macroeconomic Headwinds/Tailwinds
7. Supply Chain & Cost Efficiency
8. Customer Demand & Sentiment
9. Capital Allocation (M&A/Buybacks)
10. ESG & Public Relations (P.R.)

NEWS:
{news_text}
""".strip()




# ----------------- LOAD DATAFRAME -----------------

df = pd.read_csv("sysm/Huggingface-Brianferrell787-financial-news-multisource_2022.csv")

df = df.drop('extra_fields', axis=1)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", type=str, required=True, help="e.g. 2022-01")
args = parser.parse_args()

# Filter df to only this month
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df[df["date"].dt.to_period("M").astype(str) == args.month]
print(f"Processing month: {args.month}, rows: {len(df)}")







#df = df.iloc[6746:]
#df = df.head(5)
NEWS_COLUMN_NAME = "text" 


# prepare outputs
all_results = []
#company_name = "Tesla" 
#df_new = df.copy()
#df_new["processed"] = None

# prepare outputs
companies = ["Tesla", "Apple", "Amazon"]

# create output folder
dt = datetime.now().strftime("%Y_%m_%d")
out_dir = Path(config_params.get("saving", {}).get("output_dir", f"data_out/news_{dt}"))
out_dir.mkdir(parents=True, exist_ok=True)
#progress_path = out_dir / "Brianferrell_analyzed_progress_2022_2025_3companies_llama32_may3.csv"
#final_path = out_dir / "Brianferrell_analyzed_2022_2025_completed_3companies_llama32_may3.csv"
progress_path = out_dir / f"Brianferrell_analyzed_progress_may4{args.month}.csv"
final_path = out_dir / f"Brianferrell_analyzed_completed_may4_{args.month}.csv"


print("Started processing")
# Load existing CSV or start fresh
if final_path.exists():
    df_new = pd.read_csv(final_path)
else:
    df_new = df.copy()

# ----------------- COMPANY LOOP -----------------
for company_name in companies:
    col_name = f"processed_{company_name}"

    # Add new column for this company if not already present
    if col_name not in df_new.columns:
        df_new[col_name] = None

    #print(f"Processing company: {company_name}")

# ----------------- MAIN LOOP -----------------
for index, row in df.iterrows():
    news_text = str(row.get(NEWS_COLUMN_NAME, "")).strip()
    if not news_text:
        continue
    #print(f"Row {index}")

    # ----------------- COMPANY LOOP -----------------
    for company_name in companies:
        col_name = f"processed_{company_name}"

        # Skip if already processed for this company
        if pd.notna(df_new.at[index, col_name]):
            continue

        prompt = build_prompt(company_name, news_text)

        response = ollama.generate(
            model=llm_model,
            prompt=prompt
        )

        raw = str(response)
        m = re.search(r"response='(.*?)'\s+thinking=", raw, re.DOTALL)
        if m:
            model_text = m.group(1).strip()
        else:
            model_text = raw

        df_new.at[index, col_name] = model_text
        #print(f"  ✅ {company_name} done")


    # Save progress after every row
    df_new.to_csv(progress_path, index=False)

#print(f"✅ Done with {company_name}")

# ----------------- SAVE FINAL RESULTS -----------------
df_new.to_csv(final_path, index=False)
print(f"✅ Final results saved to {final_path}")