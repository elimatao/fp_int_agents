## Motivation
The idea for this project was to provide a question-answering interface for government data and make everyday-politics and their outcomes easier to access.

## Data
Initially, I thought about using data from [FragDenStaat](https://fragdenstaat.de/anfragen/), but i quickly realized the data was too heterogeneous to be brought into a format suitable for finetuning within a reasonable amount of effort. Then I found the page of the [Bundestag](https://dip.bundestag.de/erweiterte-suche?f.wahlperiode=20&f.herausgeber_dokumentart=Bundestag-Drucksache&f.drucksachetyp_p=04Fragen%2FAnfragen&f.drucksachetyp_p=04Fragen%2FAnfragen~Kleine%20Anfrage&rows=25) where "kleine Anfragen" can be queried and fetched via API.

The scripts in the `scripts/kleine_anfragen/` folder are used to fetch and process this data. Run them with `--help` flag for details:
1. `fetch_kleine_anfragen.py` fetches, for a given legislature period, date range and maximum number of entries, a number of inquiry metadata from the server and creates a local index. Then the pdfs with the answers to the inquiries are fetched, as they contain questions and answers. If an inquiry hasn't been answered (yet), it is skipped.
Note the API requires the use of an API key which can be gotten easily by navigating [here](https://dip.bundestag.de/%C3%BCber-dip/hilfe/api). Paste your key to `kleine_anfragen/config.toml.example` and rename the file to `config.toml`.
2. `extract_qa_pairs.py` then uses primarily the difference in font size to separate questions and answers and stores for each document all pairs in one `.jsonl` row.
3. `generate_conversation_dataset.py` flattens the documents from the previous step into simple qa pairs in standard ChatML format with user and assistant fields. The train / test / validate split can be customized.
3. `generate_reranker_dataset.py` also takes the output of step 2 to create cross-encoder reranker triplets with positive `{"query": "...", "passage": "...", "label": 1}` examples and negative ones (`"label": 0`). Negative ones are created by, within each document, shifting answers to questions by one position in round-robin fashion. This way the negative examples stay lexically related, but not semantically.

## Finetuning
In the corresponding scripts, models are first fetched from huggingface, set HF_KEY in kleine_anfragen/config.toml for faster download speeds.

The dataset used for finetuning consists of the first 300 inquries returned by the API for legislation period 20, which in total yielded 4700 qa training pairs.

### Conversation Model
For the finetuning I chose qwen2.5-3b-instruct in its mlx version without quantization as a base model. 
I trained a LoRA adapter of rank 8 with batch size 2 (the most my m4 pro with 48gb RAM could run without OOM errors). It took forever.

Results were not convincing, while the perplexity score decreased with the finetuned adapter, the quality of results is not convincing, 
which is expected as we cannot expect such a small model to store all the government facts. 
Additionally, in the data many answers consist of references to other answers that aren't useful in general without further context. 
However, I could have improved results by masking the questions during loss computation. 
But re-running full training would have wasted many resources without much real-world benefit in this case, so I decided to leave as-is.

### Reranker Model
For the reranker I chose `BAAI/bge-reranker-v2-m3` as the base cross-encoder. It was fine-tuned via `sentence-transformers` and `peft` on Apple Silicon MPS using a LoRA adapter. Further Hyperparameters:
-  Adapter rank: 8
- learning rate: `1e-4`, and a 
- maximum sequence length: 512 tokens (covered over 92% of answers without truncation).
- epochs: 2
- batches: size 8 and gradient accumulation 2
- loss function: binary cross-entropy loss

#### Results
Metric                 | Base Model   | Fine-Tuned   | Delta   
--------------------------------------------------------------
Accuracy               | 0.7260       | 0.8514       | +0.1254 
Average Precision      | 0.7709       | 0.9061       | +0.1352 
F1 Score               | 0.7262       | 0.8550       | +0.1288 
Precision              | 0.6991       | 0.8266       | +0.1274 
Recall                 | 0.7554       | 0.8854       | +0.1300 