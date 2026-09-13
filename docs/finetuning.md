## Motivation
The idea for this project was to provide a question-answering interface for government data and make everyday-politics and their outcomes easier to access.

## Data
Initially, I thought about using data from [FragDenStaat](https://fragdenstaat.de/anfragen/), but i quickly realized the data was too heterogeneous to be brought into a format suitable for finetuning within a reasonable amount of effort. Then I found the page of the [Bundestag](https://dip.bundestag.de/erweiterte-suche?f.wahlperiode=20&f.herausgeber_dokumentart=Bundestag-Drucksache&f.drucksachetyp_p=04Fragen%2FAnfragen&f.drucksachetyp_p=04Fragen%2FAnfragen~Kleine%20Anfrage&rows=25) where "kleine Anfragen" can be queried and fetched via API.

The scripts in the `scripts/kleine_anfragen/` folder are used to fetch and process this data. Run them with `--help` flag for details:
1. `fetch_kleine_anfragen.py` fetches, for a given legislature period, date range and maximum number of entries, a number of inquiry metadata from the server and creates a local index. Then the pdfs with the answers to the inquiries are fetched, as they contain questions and answers.
2. `extract_qa_pairs.py` then uses primarily the difference in font size to separate questions and answers and stores for each document all pairs in one `.jsonl` row.
3. `generate_conversation_dataset.py` flattens the documents from the previous step into simple qa pairs in standard ChatML format with user and assistant fields. The train / test / validate split can be customized.
3. `generate_reranker_dataset.py` also takes the output of step 2 to create cross-encoder reranker triplets with positive `{"query": "...", "passage": "...", "label": 1}` examples and negative ones (`"label": 0`). Negative ones are created by, within each document, shifting answers to questions by one position in round-robin fashion. This way the negative examples stay lexically related, but not semantically.

## Finetuning
### Conversation Model

### Reranker Model