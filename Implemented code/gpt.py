# -*- coding: utf-8 -*-




import openai

client = openai.OpenAI(api_key=""


)

# Notebook: changelog_feature_extraction.ipynb
# ------------------------------
# Imports & config
# ------------------------------
import os
import re
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI  # uses the OpenAI python SDK that earlier examples used
import time

LINES_LIMIT = None

# Output CSV

def read_first_n_lines(path, n=None):
    with open("A2.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    if n is not None:
        lines = lines[:n]
    # normalize: strip trailing newlines but keep blank lines as separators
    return [ln.rstrip("\n") for ln in lines]

def is_bullet_line(s):
    return bool(re.match(r'^\s*[-•*]\s+', s))

def clean_line(s):
    return s.strip()

def word_count(s):
    return len(s.split())

"""## ->Title extraction function declaration
## ->Embedding function declaration
"""

def extract_short_title(description, prefer_short_threshold=6, model="gpt-4o-mini"):
    desc = description.strip()
    if word_count(re.sub(r'[^\w\s]', '', desc)) <= prefer_short_threshold:
        # already short, return title-cased or just return
        return re.sub(r'\s+', ' ', desc)
    prompt = f"""
Extract a concise feature title from the following changelog sentence.
Return ONLY the title (2–6 words), do not add punctuation or extra commentary.

Changelog sentence:
{desc}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.0,
            max_tokens=32
        )
        title = resp.choices[0].message.content.strip()
        # sanitize newline/quotes
        title = re.sub(r'[\r\n\"\'\t]+', ' ', title).strip()

        return title
    except Exception as e:
        # fallback: last 6 words
        tokens = desc.split()
        return " ".join(tokens[:6]) if len(tokens) <= 6 else " ".join(tokens[:6])

# single-call embedding wrapper (batch)
def get_embeddings(texts, model="text-embedding-3-small", batch_size=64):
    # The SDK allows sending a list; chunk if necessary
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        embeddings.extend([r.embedding for r in resp.data])
    return np.array(embeddings)

from sklearn.decomposition import PCA
# ------------------------------
# Step A: Load text and prepare lines (first N lines)
# ------------------------------
txt_path = "changelog.txt"   # replace if needed
lines = read_first_n_lines(txt_path, n=LINES_LIMIT)
print(f"Read {len(lines)} lines (limit={LINES_LIMIT})")

# Normalize - remove leading/trailing whitespace from each line
lines = [ln.rstrip() for ln in lines]

# ------------------------------
# Step B: Build feature-description pairs
# Heuristics:
#  - If a short line (<=6 words) is followed by a longer line, treat the former as 'feature' and latter as 'description'
#  - If a bullet line is present, capture the bullet's content as a single entry
#  - If a line appears to be a sentence with '.' and there is no short-title next, treat it as a description and let GPT produce a concise title
# ------------------------------
pairs = []  # list of dicts: {raw_feature, description, combined_text}
i = 0
n = len(lines)

while i < n:
    ln = lines[i].strip()
    if ln == "":
        i += 1
        continue

    # If bullet style, capture the bullet content. Also try to capture following indented description lines.
    if is_bullet_line(ln):
        content = re.sub(r'^\s*[-•*]\s+', '', ln).strip()
        # look ahead to see if next line is a longer description (not blank and not another bullet/section header)
        desc = ""
        if (i+1 < n) and (lines[i+1].strip() != "") and (not is_bullet_line(lines[i+1])):
            # if next line is substantially longer, treat as description
            next_ln = lines[i+1].strip()
            if len(next_ln) > len(content):
                desc = next_ln
                i += 1  # skip next as we've consumed it
        pairs.append({"raw": content, "description": desc})
        i += 1
        continue

    # Not bullet: Check if this looks like a short-title followed by a descriptive sentence.
    this_count = word_count(ln)
    next_ln = lines[i+1].strip() if i+1 < n else ""
    next_count = word_count(next_ln) if next_ln else 0

    if this_count <= 6 and next_ln and not is_bullet_line(next_ln):
        # treat ln as feature title and next as description (if the next is longer or looks sentence-like)
        if next_count >= max(6, this_count + 2) or (next_ln.endswith('.') and not ln.endswith('.')):
            pairs.append({"raw": ln, "description": next_ln})
            i += 2
            continue

    # Otherwise treat the current line as a standalone description (title to be extracted by GPT)
    pairs.append({"raw": "", "description": ln})
    i += 1

print(f"Built {len(pairs)} candidate feature-description pairs (some 'raw' may be empty -> title will be extracted).")
for p in pairs:
    print("-", p["raw"] or "[no short title]", " | ", p["description"][:120])

# ------------------------------
# Step C: Ensure each pair has a feature_title (short). Use heuristic + GPT fallback
# ------------------------------
for idx, p in enumerate(pairs):
    if p["raw"]:
        # raw likely already a short title — normalize
        p["feature_title"] = p["raw"].strip()
        p["full_text"] = p["feature_title"] + (": " + p["description"] if p["description"] else "")
    else:
        # description-only: ask GPT to create a short title
        # but first do a cheap heuristic: if description is short (<=6 words) use it as title
        desc = p["description"].strip()
        if word_count(re.sub(r'[^\w\s]', '', desc)) <= 6:
            p["feature_title"] = desc
            p["full_text"] = desc
        else:
            # get short title from GPT
            title = extract_short_title(desc)
            p["feature_title"] = title
            p["full_text"] = title + ": " + desc

# Build DataFrame
df_pairs = pd.DataFrame(pairs)
df_pairs = df_pairs[["feature_title", "description", "full_text"]]
print("\nSample extracted rows:")
print(df_pairs.head(10).to_string(index=False))

import openai
import pandas as pd
import json
from tqdm import tqdm
min_words = 3
min_chars = 15
# removing descriptions less than 3 words
df_pairs = df_pairs[df_pairs['description'].apply(lambda x: isinstance(x, str) and len(x.split()) >= min_words)]
df_pairs = df_pairs[df_pairs['description'].apply(lambda x: isinstance(x, str) and len(x) >= min_chars)]



# Assuming df_pairs already exists and has a "description" column

def gpt_tag_process_object(desc):
    prompt = f"""
You are analyzing feature descriptions in software requirement text.
For the following sentence, identify whether it expresses:
- a **process**: an action or operation the system or user can perform (e.g., schedule, join, share, enable)
- an **object**: the entity being acted upon (e.g., meeting, video, chat, participant)

Return ONLY a compact JSON in this format:
{{"process": true/false, "object": true/false}}

Sentence: "{desc}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        # result = response.choices[0].message["content"].strip()
        result = response.choices[0].message.content.strip()

        data = json.loads(result)
        return data.get("process", False), data.get("object", False)
    except Exception as e:
        print("⚠️ Error:", e)
        return False, False

# Run on each description
process_list, object_list = [], []

for desc in tqdm(df_pairs["description"], desc="Tagging with GPT"):
    has_process, has_object = gpt_tag_process_object(desc)
    process_list.append(has_process)
    object_list.append(has_object)

df_pairs["has_process"] = process_list
df_pairs["has_object"] = object_list
df_pairs["has_process_object"] = df_pairs["has_process"] & df_pairs["has_object"]

print(df_pairs["has_process_object"].value_counts())
print("\nSample tagged rows:")
print(df_pairs.head(10).to_string(index=False))
output_path = "features_tagged.csv"
df_pairs.to_csv(output_path, index=False)

# ------------------------------
# Step: Remove rows without process
# ------------------------------
df_pairs = df_pairs[df_pairs["has_process"] == True].reset_index(drop=True)

# Save the filtered version
filtered_output_path = "features_tagged_filtered.csv"
df_pairs.to_csv(filtered_output_path, index=False)

print(f"\n Saved filtered dataset with only process-related rows to: {filtered_output_path}")
print(f"Remaining rows: {len(df_pairs)}")

!pip install bertopic
!pip install sentence-transformers
!pip install umap-learn hdbscan

# Ask GPT to name a cluster given a list of representative feature titles or descriptions
def name_cluster(cluster_examples, model="gpt-4o-mini"):
    # Limit length of prompt (keep to ~20 examples)
    examples = cluster_examples
    prompt = f"""
You are given a short list of feature descriptions or titles from a software changelog.
Please suggest ONE short, clear cluster name of 2-4 words that best summarizes these items.
Return only the cluster name, nothing else.

Examples:
{examples}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=16
        )
        name = resp.choices[0].message.content.strip()
        name = re.sub(r'[\r\n\"\'\t]+', ' ', name).strip()
        return name
    except Exception as e:
        # fallback simple label
        return "Miscellaneous"

import pandas as pd

# Replace 'your_file.csv' with your actual CSV file name or full path
df_pairs = pd.read_csv('/content/features_tagged_filtered.csv')



from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

# ------------------------------
# Step D: Embeddings & Clustering (Automatic using BERTopic)
# ------------------------------
texts = df_pairs["full_text"].tolist()

if len(texts) < 2:
    print("Not enough items to cluster. Writing CSV with single cluster.")
    df_pairs["cluster"] = 0
    df_pairs["cluster_name"] = "All Features"
else:
    print("Generating BERT embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, efficient BERT model
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Clustering with BERTopic...")
    topic_model = BERTopic(
        min_topic_size=2,        # ensures each cluster has at least 2 items
        calculate_probabilities=False,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)

    df_pairs["cluster"] = topics

    # ------------------------------
    # Step E: Name clusters using GPT
    # ------------------------------

# @title
def name_cluster_with_gpt(topic_words):
        keywords = ", ".join([w for w, _ in topic_words[:5]])
        prompt = f"Create a short, 2–5 word title summarizing this topic: {keywords}"

        import openai
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

cluster_names = {}
for c in sorted(df_pairs["cluster"].unique()):
    if c == -1:
        cluster_names[c] = "Miscellaneous"
        continue
    topic_words = topic_model.get_topic(c)
    name = name_cluster_with_gpt(topic_words)
    cluster_names[c] = name
    print(f"Cluster {c} → {name}")


    df_pairs["cluster_name"] = df_pairs["cluster"].map(cluster_names)

# ------------------------------
# Step F: Save CSV
# ------------------------------
import csv
import csv

# Remove any surrounding double quotes in cluster names
df_pairs["cluster_name"] = df_pairs["cluster_name"].str.replace('"', "", regex=False)

# Save normally (keep valid CSV quoting)
OUTPUT_CSV = "berttext_clustering.csv"
df_pairs.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
print(f"Saved results to: {OUTPUT_CSV}")

print("Columns in CSV: feature_title, description, full_text, cluster, cluster_name")

df_subset = df_pairs.head(30).copy()
df_subset



# @title
import pandas as pd
import random
from itertools import product
import math

new_reqs = []

for cluster, group in df_subset.groupby("cluster_name"):
    cluster_texts = "\n".join(group["description"].dropna().tolist())

    prompt = f"""
    You are a professional software requirement analyst and you are analyzing a group of software requirement statements related to the cluster: '{cluster}'.

    From the following text, extract **distinct meaningful phrases** that represent:
    - "system" → the main software name, system name, feature name or platform name (e.g., Zoom, Webex). If not present, use "software system" or "webex" as a fallback.
    - "process" → an action or operation users can perform (e.g., schedule, join, share, enable)
    - "object" → the thing being acted upon (e.g., meeting, video, chat, participants)
    - "details" → extra context or purpose (e.g., with single sign-on, across devices, with reactions)

    Text:
    ---
    {cluster_texts}
    ---

    You must output ONLY valid JSON with this exact format:
    {{
      "system": ["..."],
      "process": ["..."],
      "object": ["..."],
      "details": ["..."]
    }}
    Only use terms that exactly appear in the text above.
    """

    try:
        # Use enforced JSON response format
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        systems = data.get("system", [])
        processes = data.get("process", [])
        objects = data.get("object", [])
        details = data.get("details", [])

        if not systems or not processes or not objects:
            print(f"⚠️ Skipping cluster {cluster} (missing attributes)")
            continue


# Calculate total possible combinations
        total_combinations = len(systems) * len(processes) * len(objects) * len(details)

        # Limit number of generated requirements to avoid explosion
        if total_combinations <= 10:
            sample_size = total_combinations
        elif total_combinations <= 50:
            sample_size = math.ceil(total_combinations * 0.5)
        else:
            sample_size = min(total_combinations, 20 + len(processes))  # adaptive upper bound

        # Generate the actual combinations
        combos = list(product(systems, processes, objects, details))
        sampled_combos = combos[:sample_size]

        for s, p, o, d in sampled_combos:
            new_req = f"{s} shall provide users with the ability to {p} {o} {d}".strip()
            new_reqs.append({
                "feature_title": "",
                "description": "",
                "cluster_name": cluster,
                "system": s,
                "process": p,
                "object": o,
                "details": d,
                "boilerplate_requirement": new_req
            })

        print(f"✅ {cluster}: Generated {sample_size}/{total_combinations} boilerplate requirements.")

    except Exception as e:
        print(f"⚠️ Error for cluster {cluster}: {e}")
        continue

df_boilerplate = pd.DataFrame(new_reqs)
df_boilerplate.to_csv("boilerplate_generated_requirements_gpt.csv", index=False)
print("GPT-based boilerplate requirements saved to boilerplate_generated_requirements_gpt.csv")

import pandas as pd
import random
import math
import re
import json
from itertools import product

new_reqs = []

for cluster, group in df_pairs.groupby("cluster_name"):
    cluster_texts = "\n".join(group["description"].dropna().tolist())

    prompt = f"""
    You are a professional software requirement analyst and you are analyzing a group of software requirement statements related to the cluster: '{cluster}'.

    From the following text, extract **distinct meaningful phrases** that represent:
    - "system" → the main software name, system name, feature name or platform name (e.g., Zoom, Webex, app, meeting platform). If not present, use "software system" or "webex" as a fallback.
    - "process" → an action or operation users can perform (e.g., schedule, join, share, enable)
    - "object" → the thing being acted upon (e.g., meeting, video, chat, participants)
    - "details" → extra context or purpose (e.g., with single sign-on, across devices, with reactions)

    Text:
    ---
    {cluster_texts}
    ---

    You must output ONLY valid JSON with this exact format:
    {{
      "system": ["..."],
      "process": ["..."],
      "object": ["..."],
      "details": ["..."]
    }}
    Only use terms that exactly appear in the text above.
    """

    try:
        # GPT extraction call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        systems = data.get("system", [])
        processes = data.get("process", [])
        objects = data.get("object", [])
        details = data.get("details", [])

        if not systems or not processes or not objects:
            print(f" Skipping cluster {cluster} (missing attributes)")
            continue

        # ------------------------------------------
        # Attribute diversity rule:
        # Don’t mix two attributes from the same original requirement line.
        # ------------------------------------------
        text_lines = group["description"].dropna().tolist()
        unique_terms = [re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-\s]+\b", t.lower()) for t in text_lines]

        def came_from_same_req(a, b):
            """Return True if both attributes come from the same requirement line."""
            for terms in unique_terms:
                if a.lower() in " ".join(terms) and b.lower() in " ".join(terms):
                    return True
            return False

        all_combos = list(product(systems, processes, objects, details)) # mixing combinations #cartesian product #combinatorial mixing
        filtered_combos = []
        for s, p, o, d in all_combos:
            # Skip combinations where two or more come from same sentence
            if came_from_same_req(s, p) or came_from_same_req(p, o) or came_from_same_req(o, d):
                continue
            filtered_combos.append((s, p, o, d))

        total_combinations = len(all_combos)
        filtered_total = len(filtered_combos)

        if not filtered_combos:
            print(f" No diverse combinations found for cluster '{cluster}', generating fallback requirement.")
            s = systems[0] if systems else "software system"
            p = processes[0] if processes else "provide"
            o = objects[0] if objects else "functionality"
            d = details[0] if details else ""
            filtered_combos = [(s, p, o, d)]

        # ------------------------------------------
        # Limit number of generated requirements
        # ------------------------------------------
        if filtered_total <= 10:
            sample_size = filtered_total
        elif filtered_total <= 50:
            sample_size = math.ceil(filtered_total * 0.5)
        else:
            sample_size = min(filtered_total, 20 + len(processes))  # adaptive upper bound

        sampled_combos = random.sample(filtered_combos, min(sample_size, filtered_total))

        # ------------------------------------------
        # Create boilerplate requirements
        # ------------------------------------------
        for s, p, o, d in sampled_combos:
            new_req = f"{s} shall provide users with the ability to {p} {o} {d}".strip()
            new_reqs.append({
                "feature_title": "",
                "description": "",
                "cluster_name": cluster,
                "system": s,
                "process": p,
                "object": o,
                "details": d,
                "boilerplate_requirement": new_req
            })

        print(f"✅ {cluster}: Generated {len(sampled_combos)}/{filtered_total} diverse boilerplate requirements.")

    except Exception as e:
        print(f"⚠️ Error for cluster {cluster}: {e}")
        continue


# ------------------------------------------
# Save results
# ------------------------------------------
df_boilerplate = pd.DataFrame(new_reqs)
df_boilerplate.to_csv("boilerplate_generated_requirements_gpt.csv", index=False)
print("✅ GPT-based boilerplate requirements saved to boilerplate_generated_requirements_gpt.csv")



import pandas as pd
df_boilerplate = pd.read_csv('/content/boilerplate_generated_requirements_gpt (1).csv')
df_copy= df_boilerplate.copy()

from openai import OpenAI
import pandas as pd
import time



def to_ieee_style_with_gpt(text):
    """
    Use GPT to paraphrase a boilerplate requirement into  IEEE 1012-2024 style phrasing.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    prompt = f"""
You are an expert in IEEE 1012-2024 software requirements specification.
Rephrase the following functional requirements into a compliant IEEE format. Add some creativity on the boilerplate requirements without changing the main functionality.
Ensure the grammar and syntax are correct. Only output the rephrased version. Do not add any additional text.

Requirement:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        ieee_text = response.choices[0].message.content.strip()
        return ieee_text

    except Exception as e:
        print(f"Error: {e}")
        return text

# Load your boilerplate requirements

# Apply GPT-based IEEE transformation
df_copy["ieee_requirement"] = df_copy["boilerplate_requirement"].apply(to_ieee_style_with_gpt)

# Keep only the two desired columns
df_ieee = df_copy[["boilerplate_requirement", "ieee_requirement"]]

# Save results
df_ieee.to_csv("Final_ieee_style_requirements.csv", index=False)
print("✅ Saved IEEE-standard paraphrased requirements (2 columns only) to ieee_style_requirements.csv")

# Quick preview
print(df_ieee.head(10))



from sentence_transformers import SentenceTransformer, util
import pandas as pd
import openai
import torch
import numpy as np


# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset with columns: cluster_name, ieee_requirement
df = pd.read_csv("/content/gpt_Final_ieee_style_requirements.csv")

# --- Optional: set your OpenAI key ---
openai.api_key = ""

def get_reasonableness_score(sentence):
    """
    Ask GPT to rate the sentence for completeness, clarity, and feasibility.
    Returns a float between 0 and 1.
    """
    prompt = f"""
You are an expert software requirements analyst.
Rate the following IEEE-style requirement from 0 to 1
based on its clarity, completeness, and feasibility.
Give only a number, no explanation.

Requirement:
"{sentence}"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        score_str = response["choices"][0]["message"]["content"].strip()
        return float(score_str)
    except Exception:
        return 0.5  # neutral fallback

selected_reqs = []

for cluster, group in df.groupby("cluster_name"):
    sentences = group["ieee_requirement"].dropna().tolist()
    if len(sentences) < 2:
        selected_reqs.append(group.iloc[0])
        continue

    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    # Compute mean similarity → lower means more unique
    mean_sim = sim_matrix.mean(dim=1)
    uniqueness_scores = (1 - mean_sim.cpu().numpy())

    # LLM reasonableness scores
    quality_scores = [get_reasonableness_score(s) for s in sentences]

    # Weighted final score
    final_scores = 0.7 * uniqueness_scores + 0.3 * np.array(quality_scores)

    best_idx = final_scores.argmax()
    best_req = group.iloc[best_idx]
    best_req["uniqueness_score"] = uniqueness_scores[best_idx]
    best_req["quality_score"] = quality_scores[best_idx]
    best_req["final_score"] = final_scores[best_idx]

    selected_reqs.append(best_req)
    print(f"✅ {cluster}: {best_req['ieee_requirement']} (score={final_scores[best_idx]:.3f})")

# Combine and save
df_selected = pd.DataFrame(selected_reqs)
df_selected.to_csv("selected_high_quality_uncommon_reqs.csv", index=False)
print("✅ Saved results to selected_high_quality_uncommon_reqs.csv")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Initialize models
# Load dataset with columns: cluster_name, ieee_requirement
# df = pd.read_csv("/content/gpt_Final_ieee_style_requirements.csv")

# --- Optional: set your OpenAI key ---

client = OpenAI(api_key="")  # Replace with your actual key
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_gpt_quality_score(sentence):
    """
    Query GPT for a sentence quality score (1–10).
    You can replace this with cached scores later to save API calls.
    """
    prompt = f"""
Rate the following requirement on clarity, grammatical quality, and reasonableness on a scale of 1 to 10.
Only return the number.

Requirement: "{sentence}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return min(max(score, 1), 10)  # clip between 1 and 10
    except:
        return 5.0  # fallback default

def pick_mid_unique_high_quality(df):
    selected = []

    for cluster_id, cluster_df in tqdm(df.groupby("cluster_name"), desc="Selecting per cluster"):
        sentences = cluster_df["ieee_requirement"].tolist()
        if len(sentences) == 1:
            selected.append((cluster_id, sentences[0], 10, 10))
            continue

        # Step 1: Embeddings & similarity
        embeddings = embedding_model.encode(sentences)
        sim_matrix = cosine_similarity(embeddings)
        avg_sim = sim_matrix.mean(axis=1)
        uniqueness_scores = 1 - avg_sim  # higher = more unique

        # Step 2: GPT quality scores
        quality_scores = [get_gpt_quality_score(s) for s in sentences]

        # Step 3: Find mid-uniqueness index (around median)
        mid_idx = np.argsort(uniqueness_scores)[len(uniqueness_scores)//2]

        # Step 4: Find highest-quality requirement among ±2 around mid
        sorted_indices = np.argsort(uniqueness_scores)
        mid_window = sorted_indices[max(0, len(sorted_indices)//2 - 2): min(len(sorted_indices), len(sorted_indices)//2 + 3)]

        window_qualities = [quality_scores[i] for i in mid_window]
        best_in_window_idx = mid_window[np.argmax(window_qualities)]

        selected.append((cluster_id, sentences[best_in_window_idx],
                         uniqueness_scores[best_in_window_idx],
                         quality_scores[best_in_window_idx]))

    result_df = pd.DataFrame(selected, columns=["cluster", "selected_requirement", "uniqueness", "quality"])
    return result_df

# --- Run the selection ---
df_selected = pick_mid_unique_high_quality(df)

# Save results
df_selected.to_csv("selected_requirements_per_cluster.csv", index=False)
print("✅ Saved best balanced requirements to selected_requirements_per_cluster.csv")

"""### December 12

"""

import openai

client = openai.OpenAI(api_key=""


)

import pandas as pd
from openai import OpenAI
import time

# -------------------------------------------------------
# 1. Setup
# -------------------------------------------------------

df = pd.read_csv("/content/GPT survey req.csv")

OUTPUT_CSV = "rated_output.csv"

# -------------------------------------------------------
# 2. Helper function to rate requirement
# -------------------------------------------------------
def rate_requirement(req_text):
    prompt = f"""
You are evaluating software requirements.

Rate the following requirement on **novelty**, **usefulness**, and **clarity**.
Each rating must be a number from 1 to 5 (Likert scale):

5 = very high
4 = high
3 = medium
2 = low
1 = very low

Requirement:
\"\"\"{req_text}\"\"\"

Return ONLY JSON in this format:
{{
  "novelty": <1-5>,
  "usefulness": <1-5>,
  "clarity": <1-5>
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Parse JSON from model
    import json
    content = response.choices[0].message.content.strip()
    return json.loads(content)

# -------------------------------------------------------
# 3. Load CSV
# -------------------------------------------------------


# Ensure column exists
if "selected_requirement" not in df.columns:
    raise ValueError("CSV must contain a column named 'selected_requirements'")

# Create outputs
df["novelty"] = None
df["usefulness"] = None
df["clarity"] = None

# -------------------------------------------------------
# 4. Process each requirement
# -------------------------------------------------------
for i, row in df.iterrows():
    req = row["selected_requirement"]

    try:
        result = rate_requirement(req)
        df.at[i, "novelty"] = result["novelty"]
        df.at[i, "usefulness"] = result["usefulness"]
        df.at[i, "clarity"] = result["clarity"]

    except Exception as e:
        print(f"Error at row {i}: {e}")
        continue

    time.sleep(0.5)  # Avoid rate limits

# -------------------------------------------------------
# 5. Save output
# -------------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print("Done! Results saved to:", OUTPUT_CSV)

"""#Ordinal Krippendorff Alpha

"""

!pip install krippendorff
import pandas as pd
import krippendorff

df = pd.read_csv("survey.csv")

alpha = krippendorff.alpha(
    reliability_data=df.values,
    level_of_measurement='ordinal'
)

print(alpha)

import numpy as np
import pandas as pd

df = pd.read_csv("survey.csv")

# keep only the 5 requirement columns
X = df.iloc[:, 0:5].values  # shape: (57, 5)

# Cronbach's alpha function
def cronbach_alpha(itemscores):
    itemscores = np.array(itemscores)
    item_vars = itemscores.var(axis=0, ddof=1)
    total_var = itemscores.sum(axis=1).var(ddof=1)
    n_items = itemscores.shape[1]

    return (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)

alpha = cronbach_alpha(X)

print("Cronbach's alpha:", alpha)

df = pd.read_csv("clarity.csv")

alpha = krippendorff.alpha(
    reliability_data=df.values,
    level_of_measurement='ordinal'
)
print(df.columns),
print(alpha)

import numpy as np
import pandas as pd

df = pd.read_csv("sheet3.csv")

# keep only the 5 requirement columns
X = df.iloc[:, 0:5].values  # shape: (57, 5)

# Cronbach's alpha function
def cronbach_alpha(itemscores):
    itemscores = np.array(itemscores)
    item_vars = itemscores.var(axis=0, ddof=1)
    total_var = itemscores.sum(axis=1).var(ddof=1)
    n_items = itemscores.shape[1]

    return (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)

alpha = cronbach_alpha(X)

print("Cronbach's alpha:", alpha)

!pip install pingouin

import pandas as pd
import pingouin as pg

df = pd.read_csv("sheet3.csv")

# assume columns = Req1..Req5, rows = raters
df_long = df.reset_index().melt(id_vars="index")
df_long.columns = ["rater", "requirement", "score"]

icc = pg.intraclass_corr(
    data=df_long,
    targets="requirement",
    raters="rater",
    ratings="score"
)

print(icc)

