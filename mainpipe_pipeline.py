#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 17:16:51 2025

@author: kanepavlovich
"""
import os
import argparse
import numpy as np # 1.26.4 to be compatible with torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict
from bs4 import BeautifulSoup
import spacy 
from tokenizers import ByteLevelBPETokenizer
from detoxify import Detoxify
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from collections import Counter
import fasttext
import unicodedata
import html
import requests
import re
import json
import random
import logging
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------- Configuration / Constants ----------
DEFAULT_DATA_URL = "https://s3.us-east-1.amazonaws.com/mainpipe.maincode.com/mainpipe_data_v1.jsonl"
RANDOM_SEED = 42
MIN_CHARS = 50          # drop extremely short samples
MAX_CHARS = 10000       # drop extremely long samples
MIN_TOKEN_SIZE = 5      # Drop if there are too few tokens
TOKENIZER_VOCAB_SIZE=35_000
SHARD_SIZE = 10000000
NUM_THREADS = max(1, cpu_count() - 2)
similarity_threshold=0.85    # Threshold for fuzzy duplicate detection

RE_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Load models for nlp and embedding
spacy_nlp=spacy.load("en_core_web_sm")
LANG_MODEL = fasttext.load_model('lid.176.bin')
toxicity_model = Detoxify("original")
cluster_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load toxic terms list
df = pd.read_csv('toxic_terms.csv', header=None)
toxic_terms = df.astype(str).values.flatten().tolist()
toxic_regex = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in toxic_terms) + r")\b", 
    flags=re.IGNORECASE)

# Contact info patterns
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
phone_pattern = r'(?:(?:\+?61|0)[2-478](?:[-\s]?\d){8})'

# Code patterns
code_patterns = [
        r'\bdef\s+\w+\s*\(',             # Python function
        r'\bclass\s+\w+\s*\(',           # Python class
        r'\bimport\s+\w+',               # import statements
        r'if\s+__name__\s*==',           # Python main check
        r'\bfor\s+\w+\s+in\s+',          # loops
        r'\bwhile\s+\w+',                # while loops
        r'\breturn\b',                   # return keyword
        r'\bprint\s*\(',                 # print()
        r'=\s*["\'].*["\']',             # variable assignment with string
        r'[{}\[\];<>]',                  # symbols common in code
        r'^ {4,}\w+',                    # indented code
        r'#.*$',                         # comments
        r'```.*```',                     # markdown code block
        r'^\s*(var|let|const)\s+\w+',    # JS variable declarations
        r'<\/?[a-zA-Z]+>',               # HTML/XML tags
        r'^\s*\{.*\}\s*$',               # JSON or dict-like block
        ]
NEWLINE_PATTERN = re.compile(r'\n+')

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mainpipe")


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Utilities ----------
def download_file(url: str, dest: Path, chunk_size=1024*1024):
    logger.info("Downloading %s -> %s", url, dest)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    logger.info("Download complete")

def read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # try to recover (strip trailing commas)
                try:
                    yield json.loads(line.strip().rstrip(","))
                except Exception:
                    logger.warning("Skipping malformed line")
                    continue
                
def text_extract(record: Dict) -> str:
    # try common fields
    for key in ("text", "content", "body", "sentence", "document"):
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key].strip()
    # fallback: join string fields
    texts = [v for v in record.values() if isinstance(v, str) and v.strip()]
    return " ".join(texts)[:MAX_CHARS] if texts else ""

def is_code_snippet(text: str) -> bool:
    """
    Heuristic check to detect if text looks like source code.
    """
    if any(re.search(p, text, re.MULTILINE) for p in code_patterns):
        return True

    # Symbol density heuristic (lots of punctuation = likely code)
    symbol_chars = sum(1 for c in text if c in "{}[]();<>=_#\"'")
    symbol_density = symbol_chars / len(text)
    if symbol_density > 0.08:
        return True

    # High indentation ratio heuristic
    lines = text.splitlines()
    indented = sum(1 for l in lines if l.startswith("    ") or l.startswith("\t"))
    if len(lines) > 4 and indented / len(lines) > 0.4:
        return True

    return False

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)     # Normalize Unicode characters
    text = re.sub(r'[ \t]+', ' ', text)     # Collapse multiple spaces or tabs, but preserve newlines
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove all blank lines
    text = re.sub(r'(?m)^[ \t]+', '', text) # Remove trailing spaces
    
    return text

def clean_code(code: str) -> str:
    code = unicodedata.normalize("NFKC", code) # Normalize Unicode Characters
    code = code.replace('\r\n', '\n').replace('\r', '\n') # Normalize line endings to '\n'    
    code = re.sub(r'[ \t]+(?=\n)', '', code) # Strip trailing spaces at the end of each line but keep indentation
    code = code.strip('\n') # Remove leading/trailing blank lines (but keep internal spacing)
    code = re.sub(r'\n{2,}', '\n\n', code) # collapse excessive blank lines (>1)

    return code

def get_language_with_confidence(text, confidence_threshold=0.90):
    """Detect language using fasttext with confidence threshold"""
    text_for_detection = NEWLINE_PATTERN.sub(' ', text)  # Replace newlines with spaces
    text_for_detection = text_for_detection.strip()
    try:
        predictions = LANG_MODEL.predict(text_for_detection, k=1)
        lang = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return lang, confidence
    except:
        return "error", 0.0
    
    return lang, confidence

def make_minhash(tokens):
    m = MinHash(num_perm=128)
    for t in tokens:
        m.update(str(t).encode("utf-8"))
    return m

# ---------- Process Data ----------
def process_record(record: Dict, doc=None) -> Dict:
    """
    Perform filtering based on a single record in steps.
    1. Remove short / long text pieces
    2. Remove non-printable text
    3. Strip markdown/HTML
    4. Normalise text 
    5. Drop any non-english samples
    6. Remove Personal Identifying information with SpaCy NER.
    7. Detoxify samples 

    Inputs:
        record: dict containing at least a 'text' and 'url' field.

    Outputs:
        record: dict with updated fields including; reasons to drop, landguage, 
        and length of text
    """
    # extract text 
    text = text_extract(record)
    drop_reasons = []
    pii_hit = False
    lang = None
    confidence = 0.0
    
    # 1. Remove too short / too long pieces
    if not text:
        drop_reasons.append("empty")
        return {
            "raw_record": record,
            "text": "",
            "url": record.get("url", None),
            "len_chars": 0,
            "pii_hit": pii_hit,
            "language": lang,
            "language_confidence": confidence,
            "drop_reasons": drop_reasons,
        }

    if len(text) > MAX_CHARS:
        drop_reasons.append("too_long_chars")
    elif len(text) < MIN_CHARS:
        drop_reasons.append("too_short_chars")

    # 2. Remove non-printable text
    if not drop_reasons:
        text = ''.join(ch for ch in text if ch.isprintable() or ch in {'\n', '\t'})
        if not text.strip():
            drop_reasons.append("non_printable")

    # 3. Strip markup (HTML/Markdown)
    if not drop_reasons:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        text = re.sub(r'(```.*?```|`[^`]+`)', '', text, flags=re.S)
        text = html.unescape(text)
        
    # 4. Normalize text
    if not drop_reasons:
        if is_code_snippet(text)==True:
            text = clean_code(text)
        else:
            text = clean_text(text) 

    # 5. Filter by language
    if not drop_reasons and not is_code_snippet(text):
        try:
            lang, confidence = get_language_with_confidence(text)
            if lang != "en":
                drop_reasons.append("not_english")
            if confidence < 0.8:
                drop_reasons.append("low_language_confidence_score")
        except Exception as e:
            drop_reasons.append(f"language_detection_error: {e}")

    # 6. Remove PII (NER)
    if not drop_reasons:
        # First remove email adddresses and Australian phone numbers as SpaCy doesn't handle these
        text = re.sub(email_pattern, '[EMAIL]', text)
        text = re.sub(phone_pattern, '[PHONE]', text)
        # Then remove other PII with SpaCy
        try:
            pii_entities = ['PERSON', 'ORG', 'LOC']
            pii_found = False
            
            for ent in reversed(doc.ents):
                if ent.label_ in pii_entities:
                    text = text[:ent.start_char] + f'[{ent.label_}]' + text[ent.end_char:]
                    pii_found = True
            pii_hit = pii_found
        except Exception as e:
            drop_reasons.append(f"pii_detection_error: {e}")

    # 7. De-toxify
    if not drop_reasons:
        try:
            # cheap regex-based precheck
            if toxic_regex.search(text):
                # run Detoxify only if potentially toxic terms found
                toxicity_score = toxicity_model.predict(text)
                if any(score > 0.6 for score in toxicity_score.values()):
                    drop_reasons.append("toxic_content")
            else:
                pass
        except Exception as e:
            drop_reasons.append(f"toxicity_check_error: {e}")

    return {
        "raw_record": record,
        "text": text,
        "url": record.get("url", None),
        "len_chars": len(text),
        "pii_hit": pii_hit,
        "language": lang,
        "language_confidence": confidence,
        "drop_reasons": list(set(drop_reasons)),
    }


# ---------- Train Tokeniser ----------
def train_tokenizer(processed_records: List[Dict], outdir: Path, vocab_size: int = TOKENIZER_VOCAB_SIZE):
    """
    Train a BPE tokenizer based on the available data

    Inputs:
        records: List of dicts containing at least a 'text' field.
        outdir = Path variable where you want the tokenizer model saved.
        vocab_size = Size of the vocab for the Tokenizer

    Outputs:
        tokenizer = Tokenizer model for use on data.
    """
    logger.info("Training ByteLevel BPE tokenizer on %d samples (vocab_size=%d)", len(processed_records), vocab_size)

    tmp_txt = outdir / "tokenizer_training_data.txt"
    Path(tmp_txt).write_text(
    "\n".join(rec["text"] for rec in processed_records),
    encoding="utf-8")
            
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(tmp_txt)], vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])

    tokenizer.save_model(str(outdir))
    logger.info("Saved tokenizer to %s", outdir)
    # remove tmp file
    try:
        tmp_txt.unlink()
    except Exception:
        pass
    return tokenizer

# ---------- Tokenisation and De-duplication (Fuzzy with MinHashLSH) ----------
def tokenize_and_dedupe(records: List[Dict], tokenizer_dir, similarity_threshold=0.9, MIN_TOKEN_SIZE=5):
    """
    Calculate tokens based on BPE, and then perform de-duplication with MinHash

    Inputs:
        records: List of dicts containing at least a 'text' field.
        tokenizer_dir = Path variable where you want the tokenizer model loaded from.
        similarity_threshold = integer of the similarity threshold for de-duplication
        MIN_TOKEN_SIZE = minimum number of tokens to keep in a specified record.

    Outputs:
        Tuple containing:
            - updated_records: list of dicts with tokens and de-dupe index.
    """
    logger.info("Tokenising %d records", len(records))

    tokenizer = ByteLevelBPETokenizer(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.txt"),
    )

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)

    new_records = []
    minhashes = []
    
    # Tokenize records and create LSH index for deduping
    for i, rec in enumerate(records, start=1):
        # Work on a copy so we don't mutate the original record
        new_rec = deepcopy(rec)

        if i % 10000 == 0:
            logger.info(f"Tokenized {i}/{len(records)} records")

        text = new_rec.get("text", "")
        if not text:
            new_rec["tokens"] = []
            new_rec["n_tokens"] = 0
            new_records.append(new_rec)
            continue

        encoded = tokenizer.encode(text)
        new_rec["tokens"] = encoded.ids
        new_rec["n_tokens"] = len(encoded.ids)

        if len(encoded.ids) < MIN_TOKEN_SIZE:
            new_rec["drop_reasons"].append("too_few_tokens")

        # Build MinHash index
        mh = make_minhash(new_rec["tokens"])
        minhashes.append(mh)
        lsh.insert(str(i), mh)

        new_records.append(new_rec)

    # De-duplicate 
    logger.info("De-duplicating %d records with MinHash", len(new_records))
    seen = set()
    for i, (rec, mh) in enumerate(zip(new_records, minhashes), start=1):
        if i in seen:
            continue

        dup_ids = lsh.query(mh)
        if len(dup_ids) > 1:
            dup_ints = [int(d) for d in dup_ids]
            for did in dup_ints:
                if did != i:
                    new_records[did - 1]["drop_reasons"].append("Duplicate")
                    seen.add(did)
            seen.add(i)

    return new_records

# ---------- Cluster Text based on sentence content and identify unique group sources ----------
def cluster_text_records(records: List[Dict], model: SentenceTransformer, k_range: range = range(3, 10),
    random_state = RANDOM_SEED, show_progress_bar: bool = True) -> Tuple[List[Dict], np.ndarray, List[float], int]:
    """
    Cluster text records using sentence embeddings and KMeans.
    Automatically determines the optimal number of clusters via silhouette score.

    Inputs:
        records: List of dicts containing at least a 'text' field.
        model: SentenceTransformer model for encoding.
        k_range: Range of cluster numbers to test (default: 2–19).
        random_state: Random state for reproducibility.
        show_progress_bar: Whether to show encoding progress.

    Outputs:
        Tuple containing:
            - updated_records: list of dicts with added 'cluster' field
            - embeddings: numpy array of embeddings
            - sil_scores: list of silhouette scores for each k
            - best_k: integer, optimal number of clusters
    """
    # Generate embeddings
    texts = [rec["text"] for rec in records]
    logging.info("Generating Embeddings for Cluster analysis")
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    
    # Reduce embeddings dimensions with PCA
    pca = PCA(n_components=100, random_state=random_state)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    # Find best k using silhouette score
    logging.info("Identifying optimal number of clusters with %d workers", NUM_THREADS)
    sil_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto').fit(embeddings_reduced)
        score = silhouette_score(embeddings_reduced, kmeans.labels_, sample_size=5000)
        sil_scores.append(score)

    best_k = k_range[np.argmax(sil_scores)]
    logging.info(f"Best number of clusters (by silhouette score): {best_k}")

    # Fit KMeans with optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init='auto').fit(embeddings_reduced)
    labels = kmeans.labels_

    # Add cluster label to each record (preserve all other fields)
    updated_records = []
    for rec, label in zip(records, labels):
        updated_rec = deepcopy(rec)
        updated_rec["cluster"] = int(label)
        updated_records.append(updated_rec)

    return updated_records, embeddings_reduced, sil_scores, best_k

# ---------- Create Shards Based on Cluster assignment ----------
def create_token_limited_diverse_shards(records: List[Dict], shard_dir: Path, 
                                        prefix: str = "shard",
                                        max_tokens_per_shard = SHARD_SIZE) -> List[List[Dict]]:
    """
    Create JSON shards weighted by cluster diversity and sized with token-based limits.

    Input:
        records: List of dicts.
        max_tokens_per_shard: Max total tokens allowed per shard before starting a new one.
        shard_dir: Directory to save JSON shards .
        prefix: Filename prefix for saved shards.

    Output:
        List of shards (each shard is a list of dicts).
    """
    # Group by cluster
    clusters = defaultdict(list)
    for rec in records:
        clusters[rec["cluster"]].append(rec)
    
    # Shuffle each cluster for randomness
    for c in clusters:
        random.shuffle(clusters[c])
    
    # Calculate cluster sampling weights (proportional to cluster size)
    cluster_sizes = {c: len(recs) for c, recs in clusters.items()}
    total_size = sum(cluster_sizes.values())
    cluster_weights = {c: cluster_sizes[c] / total_size for c in clusters}

    # Create shards dynamically based on token count
    shards = []
    current_shard = []
    current_tokens = 0

    # Flatten all clusters into a weighted  sample
    cluster_ids = list(clusters.keys())

    while any(clusters.values()):  # while there are records left
        # Sample cluster according to its global weight (diversity + proportion)
        available_clusters = [c for c in cluster_ids if clusters[c]]
        probs = [cluster_weights[c] for c in available_clusters]
        total_p = sum(probs)
        probs = [p / total_p for p in probs]

        chosen_cluster = random.choices(available_clusters, weights=probs, k=1)[0]
        record = clusters[chosen_cluster].pop()

        # If adding record exceeds token limit, start a new shard
        if current_tokens + len(record["tokens"]) > max_tokens_per_shard and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_tokens = 0

        shard_record = {
            "text": record.get("text", ""),
            "tokens": record.get("tokens", []),
            "url": record.get("url", ""),
            "cluster": record.get("cluster","")
            }

        current_shard.append(shard_record)
        current_tokens += len(record["tokens"])

    # Add final shard
    if current_shard:
        shards.append(current_shard)


    # write shards to disk
    os.makedirs(shard_dir, exist_ok=True)
    for i, shard in enumerate(shards):
        path = os.path.join(shard_dir, f"{prefix}_{i+1}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(shard, f, ensure_ascii=False)
    
    logger.info(f"Saved {len(shards)} shards to {shard_dir}")
    logger.info(f"Created {len(shards)} shards with (≤ {max_tokens_per_shard:,} tokens each).")
    return shards

# ---------- Plotting ----------
def inspect_results(processed: List[Dict], tokenized_records: List[Dict],
                    clustered_records: List[Dict], plotting_dir: Path, 
                    shards: List, embeddings: np.ndarray, sil_scores: List,
                    best_k: int):
    
    os.makedirs(plotting_dir, exist_ok=True)
    os.makedirs(plotting_dir / "filtering", exist_ok=True)
    os.makedirs(plotting_dir / "clustering", exist_ok=True)
   
    # Plot the initial drop reasons
    plt.figure()
    reasons = [r for rec in processed for r in rec.get("drop_reasons", [])]
    counts = Counter(reasons)

    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=45, ha='right')
    for i, (k, v) in enumerate(counts.items()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.title(f"Drop Reasons in Total Sample ({len(processed)} records)")
    plt.tight_layout()
    plt.savefig(plotting_dir / "filtering/drop_reasons_histogram.png", dpi=300)
    
    # Plot the character length distributions
    plt.figure()
    lengths = [rec["len_chars"] for rec in processed if "len_chars" in rec]
    plt.hist(lengths, bins=1000, color='skyblue', edgecolor='black')
    plt.axvline(MIN_CHARS, color='red', linestyle='--', label=f'MIN_CHARS = {MIN_CHARS}')
    plt.axvline(MAX_CHARS, color='red', linestyle='-', label=f'MAX_CHARS = {MAX_CHARS}')
    plt.title(f"Character Length Distribution in Total Sample ({len(processed)} records)")
    plt.xlabel("Character Length")
    plt.xlim([0,30000])
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotting_dir / "filtering/char_length_hist.png", dpi=300)
    
    # Plot the proportion of PII hits.
    plt.figure()
    pii = [rec["pii_hit"] for rec in processed if "pii_hit" in rec]
    counts = pd.Series(pii).value_counts(normalize=True)
    absolute_counts = pd.Series(pii).value_counts()
    ax = counts.plot(kind="bar", color=["skyblue", "lightcoral"], edgecolor="black")
    for i, (prop, count) in enumerate(zip(counts, absolute_counts)):
        ax.text(
            i, prop + 0.01,  # position just above the bar
            f"{count}", 
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title(f"Proportion of PII Hits in Total Sample ({len(processed)} records)")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plotting_dir / "filtering/pii_hits_proportion.png", dpi=300)

    
    # Plot the Distribution of Languages
    plt.figure()
    language_counts = Counter(
    str(rec.get("language", "unknown") or "unknown") for rec in processed)
    top_15 = language_counts.most_common(15)
    langs, counts = zip(*top_15)  # unzip into two lists
    plt.bar(langs, counts, color='skyblue')
    plt.xlabel("Language")
    plt.ylabel("Count")
    plt.title("Top 15 Languages)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plotting_dir / "filtering/Language_Distribution_Top15.png", dpi=300)
    
    # Plot the drop reasons after initial filtering (duplicates and short tokens)
    plt.figure()
    drop_counts = Counter(reason for rec in tokenized_records for reason in rec.get("drop_reasons", []))
    plt.figure(figsize=(8, 4))
    bars = plt.bar(drop_counts.keys(), drop_counts.values(), color="skyblue")
    plt.title(f"Drop Reasons after initial filtering in ({len(tokenized_records)} records)")
    plt.xlabel("Reason")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), int(bar.get_height()), 
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(plotting_dir / "filtering/Drops_after_initial_filtering.png", dpi=300)
    
    # Plot the distribution of Token Length
    token_lengths = [rec["n_tokens"] for rec in clustered_records if "n_tokens" in rec]
    plt.figure()
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.title(f"Distribution of Token Lengths in final sample of {len(clustered_records)}")
    plt.xlabel("Number of Tokens (n_tokens)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plotting_dir / "token_length_distribution.png", dpi=300)

    # Plot the clusters identified with KMeans / Sentence Embedding
    plt.figure()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)    
    plt.figure(figsize=(10, 7))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                c=[rec["cluster"] for rec in clustered_records],
                cmap="tab10", alpha=0.6)
    plt.title(f"t-SNE visualization of text clusters in ({len(clustered_records)}) records")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(plotting_dir / "Clustering/Identified_Cluster_groupings.png", dpi=300)
    
    # Plot the sillhoutte scores
    plt.figure()
    plt.plot(range(1, len(sil_scores) + 1), sil_scores, marker='o', color='steelblue')
    plt.title("Silhouette Scores of cluster groups")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    x_actual = range(1, len(sil_scores) + 1)
    x_labels = range(3, 3 + len(sil_scores))
    plt.xticks(x_actual, x_labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    best_idx = sil_scores.index(max(sil_scores)) + 1
    plt.axvline(best_idx, color='red', linestyle='--', label=f'Best: {best_k} (score={max(sil_scores):.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotting_dir / "Clustering/Sillhouette_Scores.png", dpi=300)

    # Plot the distributions of Clusters in each Shard.
    plt.figure()
    cluster_counts = [Counter([rec["cluster"] for rec in shard]) for shard in shards]
    all_clusters = sorted(set().union(*[c.keys() for c in cluster_counts]))
    x = np.arange(len(cluster_counts))  
    width = 0.8 / len(all_clusters)  # adjust width to fit all clusters per shard
    for i, cluster_id in enumerate(all_clusters):
        values = [counts.get(cluster_id, 0) for counts in cluster_counts]
        plt.bar(x + i * width, values, width=width, label=f"Cluster {cluster_id}", alpha=0.85)
    plt.title("Shard Composition by Cluster")
    plt.xlabel("Shard")
    plt.ylabel("Count")
    plt.xticks(x + width * len(all_clusters) / 2, [f"Shard {i+1}" for i in range(len(cluster_counts))])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(
    title="Cluster",
    bbox_to_anchor=(1.05, 1),  # position to the right
    loc='upper left',
    borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.savefig(plotting_dir / "Clustering/Clusters_Across_Shards.png", dpi=300)

# ---------- Pipeline Orchestration ----------
def pipeline_main(input_path: Path, output_dir: Path, shard_size: int, dataset_url=None):
    """
    Run the main pipeline to process JSON data for LLM pre-training. This conducts:
        1. Loading JSON records
        2. Filtering the data for short/toxic/non-english records
        3. Trains a tokenizer on filtered data
        4. Tokenizes records and removes duplicates using MinHash
        5. Clusters records based on Sentence Embeddings
        6. Creates shards based on Cluster ID to ensure diverse shards
        7. Produces visualisations of both filtering and clustering.
        8. Saves the processed data to JSON file for training.

    Inputs:
        input_path = Path variable that specifies the location of the JSON file.
        records: List of dicts containing at least a 'text' and 'url' field.
        output_dir = Path variable that specifies where you want results saved.
        shard_size = number of MAX tokens in each shard.
    Outputs:
        clustered_records = processed and tokenized text data.
    """
    output_dir = Path(output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    shard_dir=output_dir / "shards"
    plotting_dir= output_dir / "Inspection"
    results_dir = output_dir / "Results"
    tmp_dir = output_dir / "Results"
    for d in (output_dir, tokenizer_dir, shard_dir, plotting_dir, results_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Load Records
    logger.info("Reading input JSONL %s", input_path)
    records = list(read_jsonl(input_path))
    logger.info("Loaded %d raw records", len(records))    

    # 2. Perform Base filtering in Parallel
    logger.info("Creating entity docs with SpaCy Named Entity Recognition")
    texts = [text_extract(r) for r in records]
    docs = list(tqdm(spacy_nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"], 
                                    batch_size=500, n_process=NUM_THREADS), total=len(texts), 
                                    desc="Creating spaCy docs"))
    logger.info("Processing %d records across %d workers", len(records), NUM_THREADS)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_record, r, d) for r, d in zip(records, docs)]
        processed = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]

    # Write initial processed record
    with open(tmp_dir / 'all_processed_records.jsonl', "w", encoding="utf-8") as f:
        for record in processed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    logger.info("Processing complete. Samples: %d", len(processed))
    # Remove records with non-empty drop_reasons
    processed_records = [record for record in processed if not record['drop_reasons']]
    logger.info("Processing complete. Processed: %d Samples, %d Samples remain after filtering", 
                len(processed), len(processed_records))
    
    # 3. Train Tokenizer on Processed Records
    tokenizer = train_tokenizer(processed_records, outdir=tokenizer_dir)
    
    # 4. Tokenize Records and Remove Duplicates
    logger.info("Tokenizing Records and Removing Duplicates")
    tokenized_records=tokenize_and_dedupe(processed_records, tokenizer_dir=tokenizer_dir)
    tokenized_deduped_records = [record for record in tokenized_records if not record['drop_reasons']]
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable parallised tokenizer so that it doesn't interfer with clustering parallel

    # 5. Cluster Records with Sentence Embeddings & Kmeans
    clustered_records, embeddings, sil_scores, best_k = cluster_text_records(
        tokenized_deduped_records, model=cluster_embedding_model,  k_range=range(3, 10))
    
    # 6. Create Shards where split is performed based on Cluster Assignment for Diverse samples.
    shards = create_token_limited_diverse_shards(clustered_records, shard_dir=shard_dir,
                                                 max_tokens_per_shard=SHARD_SIZE)

    # 7. Produce Inspection Visualisations for reports
    inspect_results(processed, tokenized_records, clustered_records, plotting_dir = plotting_dir,
                    shards=shards, embeddings=embeddings, sil_scores=sil_scores, best_k=best_k)

    # 8. Save final JSON file after processing.
    all_records_path = results_dir / "processed_records_full.jsonl"
    tokens_records_path = results_dir / "processed_records_for_training.jsonl"
    with open(all_records_path, "w", encoding="utf-8") as f:
        for record in clustered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    fields_to_keep = {"text", "tokens", "url"}
    filtered_records = [
        {k: v for k, v in record.items() if k in fields_to_keep}
        for record in clustered_records]

    with open(tokens_records_path, "w", encoding="utf-8") as f:
        for record in filtered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    return clustered_records

# ---------- Command Line Interface ----------
def parse_args():
    p = argparse.ArgumentParser(description="Mainpipe Data Engineer Pipeline Assignment")
    p.add_argument("--input", type=str, default=None, help="Input JSONL path. If omitted and --download set, will download dataset.")
    p.add_argument("--download", action="store_true", help="Download dataset from canonical URL")
    p.add_argument("--url", type=str, default=DEFAULT_DATA_URL, help="URL to download dataset from")
    p.add_argument("--output_dir", type=str, default="output", help="Output directory")
    p.add_argument("--shard_size", type=int, default=SHARD_SIZE, help="Tokens per shard")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    # determine input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"Input path {input_path} not found")
    else:
        if not args.download:
            raise SystemExit("Either --input or --download must be provided")
        input_path = outdir / "mainpipe_data_v1.jsonl"
        if not input_path.exists():
            download_file(args.url, input_path)

    final_records = pipeline_main(input_path, outdir, shard_size=args.shard_size)
    logger.info("Done. See %s for outputs", outdir)


if __name__ == "__main__":
    main()



