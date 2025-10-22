<h1 align="center">🧠 MainPipe Pipeline </h1>

<p align="center">
  <em>An end-to-end data processing and analysis pipeline for the Maincode Take Home assessment.</em>
</p>

<hr>

<h2>📖 Introduction</h2>

<p>
  This pipeline handles data acquisition, preprocessing, tokenization, diverse shard creation, and inspection visualisations — all packaged in a Docker container.
</p>

<p>
  The schematic below provides an overview of the pipeline’s architecture and flow:
</p>

<p align="center">
  <img src="Pipeline_schematic.jpg" alt="Pipeline schematic" width="700">
</p>

<hr>

<h2>🧩 Overview</h2>

<p>
  To run this pipeline please download the Docker Image provided below 
  I have also provided a downloadable zip of the full output produced by the pipeline at the link below.
</p>

<ul>
  <li>📦 <strong>Docker Image:</strong> <a href="https://drive.google.com/file/d/14y_K4tn3fFyw1zqXtV4lgguwXH0lk5Zi/view?usp=sharing" target="_blank">Download mainpipe_kane_pavlovich.tar</a></li>
  <li>📊 <strong>Full Output:</strong> <a href="https://drive.google.com/file/d/1t9jqFVa7J083EmAk7-6H8GY5hpxGCt4i/view?usp=sharing" target="_blank">View processed results</a></li>
</ul>

<hr>

<h2>📦 Loading the Docker Image</h2>

<p>
  Once you’ve downloaded the Docker image file, load it into your local Docker environment using:
</p>

<pre><code>docker load -i mainpipe_kane_pavlovich.tar</code></pre>

<hr>

<h2>🚀 Running the Pipeline</h2>

<p>
  Open a terminal in the directory containing this repository and run one of the following commands depending on your setup.
</p>

<h3>1️⃣ Download the dataset and run the full analysis</h3>

<pre><code>docker run -v "$(pwd):/data" mainpipe_kane_pavlovich \
  python mainpipe_pipeline.py --download --output_dir /data/output
</code></pre>

<h3>2️⃣ Run with a local dataset (already downloaded into the local directory)</h3>

<pre><code>docker run -v "$(pwd):/data" mainpipe_kane_pavlovich \
  python mainpipe_pipeline.py --input /data/mainpipe_data_v1.jsonl --output_dir /data/output
</code></pre>

<hr>

<h2>⏱️ Runtime and Outputs</h2>

<p>
  The full pipeline takes approximately <strong>4 hours</strong> to complete. 
  For convenience, a truncated dataset is provided for quick testing, allowing you to see the code run without the full processing time.
</p>

<h3>⚡ Quick Test (Truncated Data)</h3>

<pre><code>docker run -v "$(pwd):/data" mainpipe_kane_pavlovich \
  python mainpipe_pipeline.py --input /data/truncated_records.jsonl --output_dir /data/truncated_output
</code></pre>

<hr>
