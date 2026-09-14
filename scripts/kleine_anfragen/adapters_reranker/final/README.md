---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:9392
- loss:BinaryCrossEntropyLoss
base_model: BAAI/bge-reranker-v2-m3
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on BAAI/bge-reranker-v2-m3
  results:
  - task:
      type: cross-encoder-classification
      name: Cross Encoder Classification
    dataset:
      name: val
      type: val
    metrics:
    - type: accuracy
      value: 0.8515901060070671
      name: Accuracy
    - type: accuracy_threshold
      value: 0.4348752796649933
      name: Accuracy Threshold
    - type: f1
      value: 0.8573797678275291
      name: F1
    - type: f1_threshold
      value: 0.32942160964012146
      name: F1 Threshold
    - type: precision
      value: 0.8078125
      name: Precision
    - type: recall
      value: 0.9134275618374559
      name: Recall
    - type: average_precision
      value: 0.9393065417758889
      name: Average Precision
---

# CrossEncoder based on BAAI/bge-reranker-v2-m3

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) <!-- at revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Wie bringt die Bundesregierung konkret die flächendeckende Digitalisierung der Schiene bundesweit voran?', 'Die Digitalisierung des Schienenverkehrs ist nach Auffassung des Bundesministeriums für Digitales und Verkehr (BMDV) eine entscheidende Maßnahme zur Modernisierung und Effizienzsteigerung des deutschen Schienennetzes. Eine abgängige Alttechnik, abgekündigte Technologien und europäische Verpflichtungen erfordern eine zeitnahe Einführung der digitalen Technologien. Das BMDV bringt die Digitalisierung des Schienennetzes beispielsweise durch laufende Projekte der ETCS-Ausrüstung, wie den Digitalen Knoten Stuttgart (DKS), den Korridor Rhein-Alpen oder im Zuge der Generalsanierung der Hochleistungskorridore sowie durch die Finanzierung des Digitalen Kapazitätsmanagements voran. So konnten etwa mit dem Haushalt 2024 durch den Abschluss von 10 Finanzierungsvereinbarungen bzw. Änderungsvereinbarungen zu laufenden Maßnahmen für die Vorhaben der Digitalen Schiene rd. 2,3 Mrd. Euro Haushaltsmittel und Verpflichtungsermächtigungen zusätzlich gebunden werden. Dabei wurden auch die durch die BSWAG-Novelle in 2024 erweiterten Finanzierungsmöglichkeiten genutzt. Weitere Mittel werden in Form einer Eigenkapitalerhöhung der Deutschen Bahn AG (DB AG) in 2025 für ihr Programm Digitale Schiene Deutschland (DSD) verfügbar gemacht, davon allein knapp 350 Mio. Euro zusätzlich für die ERTMS-Ausrüstung des Rhein-Alpen-Korridors. Damit bekennt sich der Bund klar zu seiner Finanzierungsverantwortung (vgl. auch Antwort zu Frage 2). Ausgehend von den Ergebnissen der aktualisierten Machbarkeitsstudie „Neuausrichtung der Gesamtstrategie zur Digitalisierung der Schiene“ (im Folgenden: Machbarkeitsstudie) befasst sich das BMDV darüber hinaus mit der Erarbeitung eines ganzheitlichen Konzepts zur strategischen Steuerung des Programms zur Digitalisierung der Schiene. Kernbestandteile dieses Konzepts sind eine stärkere Rolle des Bundes in der strategischen Steuerung der Digitalisierung und der Aufbau einer operativen Lenkungsinstitution gemeinsam mit dem Sektor. Bezüglich der Fahrzeugausrüstung wird neben dem bereits bestehenden Modellvorhaben DKS derzeit die Förderrichtlinie für ein First-of-Class-Sofortprogramm erarbeitet.'],
    ['Wie bringt die Bundesregierung konkret die flächendeckende Digitalisierung der Schiene bundesweit voran?', 'Die Aufwendungen von Bund und DB AG für die Digitalisierung der Schiene können der folgenden Tabelle entnommen werden. [in Mio. Euro] 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024** Bundesmittel 5 1 5 13 29 161 316 159 213 357 Eigenkapitalerhöhung* 33 71 30 67 65 Mittel DB AG 2 22 18 35 62 112 83 225 200 Die kaufmännischen Systeme der DB AG orientieren sich an einer projektbasierten Struktur und nicht an Innovationsfeldern. Eine Aufteilung im Sinne der Fragestellung liegt nicht vor. Alle Mittel werden vollständig in die Realisierung der Projekte eingebracht. Die Vergabequote im Rahmen des Wettbewerbsverfahrens lag 2024 bei 82,5 Prozent für die Projekte der Digitalisierung der Schiene.'],
    ['Wie viele Mittel haben Bund und DB AG insgesamt seit Beginn der Förderung von ETCS im Jahr 2015 (https://bmdv.bund.de/SharedDocs/DE/ Artikel/E/digitalisierung-der-schiene.html) für die Digitalisierung der Schiene aufgewendet (bitte einzeln pro Jahr auflisten)? a) Wie verteilen sich diese Mittel gemäß den Innovationsfeldern des Programms „Digitale Schiene Deutschland“. Hierzu zählen ETCS, DSTW, iLBS [integriertes Leitund Bediensystem], ATO [Automatic Train Operation] GoA [Grade of Automation] 2, CTMS [Capacity & Traffic Management System], ATO GoA 4, ADI [Advanced Digital Infrastructure] und FRMCS [Future Railway Mobile Communication System] (https://digitale-schiene-deutschland.de/Vision-und-Ziel bild)? b) Wie viele der Mittel fließen in Projektrealisierungen und wie viele anteilig in Vergaben in Wettbewerbsverfahren? * Eigenkapitalerhöhung im Zusammenhang mit dem Klimaschutzprogramm 2030. ** Vorläufige Daten für das Jahr 2024.', 'Die Aufwendungen von Bund und DB AG für die Digitalisierung der Schiene können der folgenden Tabelle entnommen werden. [in Mio. Euro] 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024** Bundesmittel 5 1 5 13 29 161 316 159 213 357 Eigenkapitalerhöhung* 33 71 30 67 65 Mittel DB AG 2 22 18 35 62 112 83 225 200 Die kaufmännischen Systeme der DB AG orientieren sich an einer projektbasierten Struktur und nicht an Innovationsfeldern. Eine Aufteilung im Sinne der Fragestellung liegt nicht vor. Alle Mittel werden vollständig in die Realisierung der Projekte eingebracht. Die Vergabequote im Rahmen des Wettbewerbsverfahrens lag 2024 bei 82,5 Prozent für die Projekte der Digitalisierung der Schiene.'],
    ['Wie viele Mittel haben Bund und DB AG insgesamt seit Beginn der Förderung von ETCS im Jahr 2015 (https://bmdv.bund.de/SharedDocs/DE/ Artikel/E/digitalisierung-der-schiene.html) für die Digitalisierung der Schiene aufgewendet (bitte einzeln pro Jahr auflisten)? a) Wie verteilen sich diese Mittel gemäß den Innovationsfeldern des Programms „Digitale Schiene Deutschland“. Hierzu zählen ETCS, DSTW, iLBS [integriertes Leitund Bediensystem], ATO [Automatic Train Operation] GoA [Grade of Automation] 2, CTMS [Capacity & Traffic Management System], ATO GoA 4, ADI [Advanced Digital Infrastructure] und FRMCS [Future Railway Mobile Communication System] (https://digitale-schiene-deutschland.de/Vision-und-Ziel bild)? b) Wie viele der Mittel fließen in Projektrealisierungen und wie viele anteilig in Vergaben in Wettbewerbsverfahren? * Eigenkapitalerhöhung im Zusammenhang mit dem Klimaschutzprogramm 2030. ** Vorläufige Daten für das Jahr 2024.', 'Für die Jahre 2025 bis 2029 ist im Bereich des Bedarfsplans Schiene von einem Finanzbedarf in Höhe von rund 14 Mrd. Euro auszugehen. Die 2024/2025 erfolgte Absenkung der mittelfristigen Finanzlinie hat allerdings bereits zu einem Investitionsstau geführt, der auch bei einer vollständigen Abdeckung des vorgenannten Finanzbedarfs Repriorisierungen unumgänglich macht. Für die nächsten fünf Jahre sind im Bereich der Digitalisierung in dem Kapitel 1202 Titel 891 06 „Ausrüstung der deutschen Infrastruktur und von rollendem Material mit dem Europäischen Zugsicherungssystem ERTMS (European Rail Traffic Management System)“ über 3,5 Mrd. Euro vertraglich gebunden. In der o. g. Machbarkeitsstudie wird von den Gutachtern des Bundes ein Finanzbedarf an Bundesmitteln in Höhe von etwa 8 Mrd. Euro bis 2029 ermittelt.'],
    ['Wie hoch ist der Finanzbedarf zur bisher geplanten Umsetzung des Bedarfsplans Schiene sowie zu der Digitalisierung (unter Ausschluss aller Gremienvorbehalte sowie unter Ausschluss aller Repriorisierungen) jeweils in den nächsten fünf Jahren?', 'Für die Jahre 2025 bis 2029 ist im Bereich des Bedarfsplans Schiene von einem Finanzbedarf in Höhe von rund 14 Mrd. Euro auszugehen. Die 2024/2025 erfolgte Absenkung der mittelfristigen Finanzlinie hat allerdings bereits zu einem Investitionsstau geführt, der auch bei einer vollständigen Abdeckung des vorgenannten Finanzbedarfs Repriorisierungen unumgänglich macht. Für die nächsten fünf Jahre sind im Bereich der Digitalisierung in dem Kapitel 1202 Titel 891 06 „Ausrüstung der deutschen Infrastruktur und von rollendem Material mit dem Europäischen Zugsicherungssystem ERTMS (European Rail Traffic Management System)“ über 3,5 Mrd. Euro vertraglich gebunden. In der o. g. Machbarkeitsstudie wird von den Gutachtern des Bundes ein Finanzbedarf an Bundesmitteln in Höhe von etwa 8 Mrd. Euro bis 2029 ermittelt.'],
]
scores = model.predict(pairs)
print(scores)
# [0.9635 0.0408 0.997  0.0347 0.9973]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Wie bringt die Bundesregierung konkret die flächendeckende Digitalisierung der Schiene bundesweit voran?',
    [
        'Die Digitalisierung des Schienenverkehrs ist nach Auffassung des Bundesministeriums für Digitales und Verkehr (BMDV) eine entscheidende Maßnahme zur Modernisierung und Effizienzsteigerung des deutschen Schienennetzes. Eine abgängige Alttechnik, abgekündigte Technologien und europäische Verpflichtungen erfordern eine zeitnahe Einführung der digitalen Technologien. Das BMDV bringt die Digitalisierung des Schienennetzes beispielsweise durch laufende Projekte der ETCS-Ausrüstung, wie den Digitalen Knoten Stuttgart (DKS), den Korridor Rhein-Alpen oder im Zuge der Generalsanierung der Hochleistungskorridore sowie durch die Finanzierung des Digitalen Kapazitätsmanagements voran. So konnten etwa mit dem Haushalt 2024 durch den Abschluss von 10 Finanzierungsvereinbarungen bzw. Änderungsvereinbarungen zu laufenden Maßnahmen für die Vorhaben der Digitalen Schiene rd. 2,3 Mrd. Euro Haushaltsmittel und Verpflichtungsermächtigungen zusätzlich gebunden werden. Dabei wurden auch die durch die BSWAG-Novelle in 2024 erweiterten Finanzierungsmöglichkeiten genutzt. Weitere Mittel werden in Form einer Eigenkapitalerhöhung der Deutschen Bahn AG (DB AG) in 2025 für ihr Programm Digitale Schiene Deutschland (DSD) verfügbar gemacht, davon allein knapp 350 Mio. Euro zusätzlich für die ERTMS-Ausrüstung des Rhein-Alpen-Korridors. Damit bekennt sich der Bund klar zu seiner Finanzierungsverantwortung (vgl. auch Antwort zu Frage 2). Ausgehend von den Ergebnissen der aktualisierten Machbarkeitsstudie „Neuausrichtung der Gesamtstrategie zur Digitalisierung der Schiene“ (im Folgenden: Machbarkeitsstudie) befasst sich das BMDV darüber hinaus mit der Erarbeitung eines ganzheitlichen Konzepts zur strategischen Steuerung des Programms zur Digitalisierung der Schiene. Kernbestandteile dieses Konzepts sind eine stärkere Rolle des Bundes in der strategischen Steuerung der Digitalisierung und der Aufbau einer operativen Lenkungsinstitution gemeinsam mit dem Sektor. Bezüglich der Fahrzeugausrüstung wird neben dem bereits bestehenden Modellvorhaben DKS derzeit die Förderrichtlinie für ein First-of-Class-Sofortprogramm erarbeitet.',
        'Die Aufwendungen von Bund und DB AG für die Digitalisierung der Schiene können der folgenden Tabelle entnommen werden. [in Mio. Euro] 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024** Bundesmittel 5 1 5 13 29 161 316 159 213 357 Eigenkapitalerhöhung* 33 71 30 67 65 Mittel DB AG 2 22 18 35 62 112 83 225 200 Die kaufmännischen Systeme der DB AG orientieren sich an einer projektbasierten Struktur und nicht an Innovationsfeldern. Eine Aufteilung im Sinne der Fragestellung liegt nicht vor. Alle Mittel werden vollständig in die Realisierung der Projekte eingebracht. Die Vergabequote im Rahmen des Wettbewerbsverfahrens lag 2024 bei 82,5 Prozent für die Projekte der Digitalisierung der Schiene.',
        'Die Aufwendungen von Bund und DB AG für die Digitalisierung der Schiene können der folgenden Tabelle entnommen werden. [in Mio. Euro] 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024** Bundesmittel 5 1 5 13 29 161 316 159 213 357 Eigenkapitalerhöhung* 33 71 30 67 65 Mittel DB AG 2 22 18 35 62 112 83 225 200 Die kaufmännischen Systeme der DB AG orientieren sich an einer projektbasierten Struktur und nicht an Innovationsfeldern. Eine Aufteilung im Sinne der Fragestellung liegt nicht vor. Alle Mittel werden vollständig in die Realisierung der Projekte eingebracht. Die Vergabequote im Rahmen des Wettbewerbsverfahrens lag 2024 bei 82,5 Prozent für die Projekte der Digitalisierung der Schiene.',
        'Für die Jahre 2025 bis 2029 ist im Bereich des Bedarfsplans Schiene von einem Finanzbedarf in Höhe von rund 14 Mrd. Euro auszugehen. Die 2024/2025 erfolgte Absenkung der mittelfristigen Finanzlinie hat allerdings bereits zu einem Investitionsstau geführt, der auch bei einer vollständigen Abdeckung des vorgenannten Finanzbedarfs Repriorisierungen unumgänglich macht. Für die nächsten fünf Jahre sind im Bereich der Digitalisierung in dem Kapitel 1202 Titel 891 06 „Ausrüstung der deutschen Infrastruktur und von rollendem Material mit dem Europäischen Zugsicherungssystem ERTMS (European Rail Traffic Management System)“ über 3,5 Mrd. Euro vertraglich gebunden. In der o. g. Machbarkeitsstudie wird von den Gutachtern des Bundes ein Finanzbedarf an Bundesmitteln in Höhe von etwa 8 Mrd. Euro bis 2029 ermittelt.',
        'Für die Jahre 2025 bis 2029 ist im Bereich des Bedarfsplans Schiene von einem Finanzbedarf in Höhe von rund 14 Mrd. Euro auszugehen. Die 2024/2025 erfolgte Absenkung der mittelfristigen Finanzlinie hat allerdings bereits zu einem Investitionsstau geführt, der auch bei einer vollständigen Abdeckung des vorgenannten Finanzbedarfs Repriorisierungen unumgänglich macht. Für die nächsten fünf Jahre sind im Bereich der Digitalisierung in dem Kapitel 1202 Titel 891 06 „Ausrüstung der deutschen Infrastruktur und von rollendem Material mit dem Europäischen Zugsicherungssystem ERTMS (European Rail Traffic Management System)“ über 3,5 Mrd. Euro vertraglich gebunden. In der o. g. Machbarkeitsstudie wird von den Gutachtern des Bundes ein Finanzbedarf an Bundesmitteln in Höhe von etwa 8 Mrd. Euro bis 2029 ermittelt.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Classification

* Dataset: `val`
* Evaluated with [<code>CrossEncoderClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.8516     |
| accuracy_threshold    | 0.4349     |
| f1                    | 0.8574     |
| f1_threshold          | 0.3294     |
| precision             | 0.8078     |
| recall                | 0.9134     |
| **average_precision** | **0.9393** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 9,392 training samples
* Columns: <code>sentence_A</code>, <code>sentence_B</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_A                                                                           | sentence_B                                                                           | label                                                         |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               | float                                                         |
  | modality | text                                                                                 | text                                                                                 |                                                               |
  | details  | <ul><li>min: 16 tokens</li><li>mean: 122.98 tokens</li><li>max: 370 tokens</li></ul> | <ul><li>min: 14 tokens</li><li>mean: 217.44 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Gab es Treffen von Vertretern der Deutschen Bahn mit Vertretern des Bundeskanzleramtes, des Bundesministeriums für Wirtschaft und Klimaschutz, des Bundesministeriums für Arbeit und Soziales sowie des Bundesministeriums für Digitales und Verkehr in der 20. Legislaturperiode, die die wirtschaftliche Entwicklung des Unternehmens sowie den geplanten Stellenabbau zum Gesprächsgegenstand hatten? a) Wenn ja, wann, und zwischen welchen Vertretern des Unternehmens sowie der genannten Einheiten der Bundesregierung fanden diese Treffen jeweils statt? b) Wenn ja, welche Vorschläge bzw. Forderungen haben die Vertreter des Unternehmens den genannten Einheiten der Bundesregierung dabei unterbreitet? c) Wenn ja, haben Vertreter der oben genannten Einheiten der Bundesregierung dabei dem Unternehmen Zusagen oder das Ergreifen von Maßnahmen (bzw. Förderung, Kurzarbeitergeld etc.) in Aussicht gestellt, und wenn ja, welche Vertreter der genannten Einheiten der Bundesregierung haben jeweils welche Zusagen e...</code> | <code>Die Fragen 1 bis 1c werden gemeinsam beantwortet. Der nachfolgenden Übersicht können die erfolgten Treffen mit Vertretern der Deutschen Bahn und Vertretern des Bundesministeriums für Wirtschaft und Klimaschutz, des Bundesministeriums für Digitales und Verkehr sowie des Bundesministeriums für Arbeit und Soziales entnommen werden. Datum Vertreter der Vertreter Deutsche Bahn Ggf. Forderungen Unternehmen bzw. Bundesregierung Zusagen/Ankündigungen der BReg/ Anlass Treffen des Bundeskanzleramtes 19.09.2024 Sts Kukies a. D. u. a. Hr. Dr. Holle 15.–17.09.2024 BK Hr. Warbamoff Mitreise auf BK – Reise nach Usbekistan, Kasachstan 31.05.2024 Sts Kukies a. D. Hr. Dr. Holle 25.03.2024 ChefBK Hr. Dr. Holle 11.01.2024 BK u. a. Hr. Dr. Lutz, Fr. Eröffnung des neuen Fahrzeuginstand- Dr. Gerd tom Markotten haltungswerks der DB AG in Cottbus 31.08.2023 ChefBK Hr. Dr. Holle 04.07.2023 Sts Kukies a. D. Hr. Dr. Holle 02.05.2023 Sts Kukies a. D. Hr. Dr. Holle 03.02.2023 ChefBK Hr. Dr. Lutz, V 17.11.2022 StM S...</code> | <code>1.0</code> |
  | <code>Gab es Treffen von Vertretern der Deutschen Bahn mit Vertretern des Bundeskanzleramtes, des Bundesministeriums für Wirtschaft und Klimaschutz, des Bundesministeriums für Arbeit und Soziales sowie des Bundesministeriums für Digitales und Verkehr in der 20. Legislaturperiode, die die wirtschaftliche Entwicklung des Unternehmens sowie den geplanten Stellenabbau zum Gesprächsgegenstand hatten? a) Wenn ja, wann, und zwischen welchen Vertretern des Unternehmens sowie der genannten Einheiten der Bundesregierung fanden diese Treffen jeweils statt? b) Wenn ja, welche Vorschläge bzw. Forderungen haben die Vertreter des Unternehmens den genannten Einheiten der Bundesregierung dabei unterbreitet? c) Wenn ja, haben Vertreter der oben genannten Einheiten der Bundesregierung dabei dem Unternehmen Zusagen oder das Ergreifen von Maßnahmen (bzw. Förderung, Kurzarbeitergeld etc.) in Aussicht gestellt, und wenn ja, welche Vertreter der genannten Einheiten der Bundesregierung haben jeweils welche Zusagen e...</code> | <code>Die Fragen 2 bis 2c werden gemeinsam beantwortet. Der nachfolgenden Übersicht können die erfolgten Treffen mit Vertretern von ZF und Vertretern des Bundeskanzleramtes sowie des Bundesministeriums für Wirtschaft und Klimaschutz entnommen werden: Datum Vertreter Vertreter ZF Ggf. Forderungen Un­ der Bundes­ ternehmen bzw. Zusa­ regierung gen/Ankündigungen der BReg/Anlass Treffen des Bundeskanzleramtes 09.10.2024 Sts Kukies Hr. Dr. Klein, CEO ZF Group/ZF Friedrichshafen a. D. AG 27.11.2023 BK, ChefBK u. a. Hr. Dr. Klein, Hr. Dietrich (Gesamtbetriebsrat- 2. Spitzengespräch der Vorsitzender) Strategieplattform Transformation der Automobilund Mobilitätswirtschaft 23.11.2023 Sts Kukies Hr. Dr. Klein a. D. 13.06.2023 Sts Kukies Hr. Dietrich (Gesamtbetriebsrat-Vorsitzender) a. D. Datum Vertreter Vertreter ZF Ggf. Forderungen Un­ der Bundes­ ternehmen bzw. Zusa­ regierung gen/Ankündigungen der BReg/Anlass 10.01.2023 BK, ChefBK u. a. Hr. Dr. Klein, Hr. Dietrich (Gesamtbetriebsrat- 1. Spitzengespr...</code> | <code>0.0</code> |
  | <code>Gab es Treffen von Vertretern von ZF mit Vertretern des Bundeskanzleramtes, des Bundesministeriums für Wirtschaft und Klimaschutz, des Bundesministeriums für Arbeit und Soziales sowie des Bundesministeriums für Digitales und Verkehr in der 20. Legislaturperiode, die die wirtschaftliche Entwicklung des Unternehmens sowie den geplanten Stellenabbau zum Gesprächsgegenstand hatten? a) Wenn ja, wann, und zwischen welchen Vertretern des Unternehmens sowie der genannten Einheiten der Bundesregierung fanden diese Treffen jeweils statt? b) Wenn ja, welche Vorschläge bzw. Forderungen haben die Vertreter des Unternehmens den genannten Einheiten der Bundesregierung dabei unterbreitet? c) Wenn ja, haben Vertreter der oben genannten Einheiten der Bundesregierung dabei dem Unternehmen Zusagen oder das Ergreifen von Maßnahmen (bzw. Förderung, Kurzarbeitergeld etc.) in Aussicht gestellt, und wenn ja, welche Vertreter der genannten Einheiten der Bundesregierung haben jeweils welche Zusagen erteilt bzw. ...</code> | <code>Die Fragen 2 bis 2c werden gemeinsam beantwortet. Der nachfolgenden Übersicht können die erfolgten Treffen mit Vertretern von ZF und Vertretern des Bundeskanzleramtes sowie des Bundesministeriums für Wirtschaft und Klimaschutz entnommen werden: Datum Vertreter Vertreter ZF Ggf. Forderungen Un­ der Bundes­ ternehmen bzw. Zusa­ regierung gen/Ankündigungen der BReg/Anlass Treffen des Bundeskanzleramtes 09.10.2024 Sts Kukies Hr. Dr. Klein, CEO ZF Group/ZF Friedrichshafen a. D. AG 27.11.2023 BK, ChefBK u. a. Hr. Dr. Klein, Hr. Dietrich (Gesamtbetriebsrat- 2. Spitzengespräch der Vorsitzender) Strategieplattform Transformation der Automobilund Mobilitätswirtschaft 23.11.2023 Sts Kukies Hr. Dr. Klein a. D. 13.06.2023 Sts Kukies Hr. Dietrich (Gesamtbetriebsrat-Vorsitzender) a. D. Datum Vertreter Vertreter ZF Ggf. Forderungen Un­ der Bundes­ ternehmen bzw. Zusa­ regierung gen/Ankündigungen der BReg/Anlass 10.01.2023 BK, ChefBK u. a. Hr. Dr. Klein, Hr. Dietrich (Gesamtbetriebsrat- 1. Spitzengespr...</code> | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 1,132 evaluation samples
* Columns: <code>sentence_A</code>, <code>sentence_B</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_A                                                                           | sentence_B                                                                           | label                                                         |
  |:---------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type     | string                                                                               | string                                                                               | float                                                         |
  | modality | text                                                                                 | text                                                                                 |                                                               |
  | details  | <ul><li>min: 19 tokens</li><li>mean: 127.19 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 202.63 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | sentence_B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Wie bringt die Bundesregierung konkret die flächendeckende Digitalisierung der Schiene bundesweit voran?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | <code>Die Digitalisierung des Schienenverkehrs ist nach Auffassung des Bundesministeriums für Digitales und Verkehr (BMDV) eine entscheidende Maßnahme zur Modernisierung und Effizienzsteigerung des deutschen Schienennetzes. Eine abgängige Alttechnik, abgekündigte Technologien und europäische Verpflichtungen erfordern eine zeitnahe Einführung der digitalen Technologien. Das BMDV bringt die Digitalisierung des Schienennetzes beispielsweise durch laufende Projekte der ETCS-Ausrüstung, wie den Digitalen Knoten Stuttgart (DKS), den Korridor Rhein-Alpen oder im Zuge der Generalsanierung der Hochleistungskorridore sowie durch die Finanzierung des Digitalen Kapazitätsmanagements voran. So konnten etwa mit dem Haushalt 2024 durch den Abschluss von 10 Finanzierungsvereinbarungen bzw. Änderungsvereinbarungen zu laufenden Maßnahmen für die Vorhaben der Digitalen Schiene rd. 2,3 Mrd. Euro Haushaltsmittel und Verpflichtungsermächtigungen zusätzlich gebunden werden. Dabei wurden auch die durch die BSWAG-No...</code> | <code>1.0</code> |
  | <code>Wie bringt die Bundesregierung konkret die flächendeckende Digitalisierung der Schiene bundesweit voran?</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | <code>Die Aufwendungen von Bund und DB AG für die Digitalisierung der Schiene können der folgenden Tabelle entnommen werden. [in Mio. Euro] 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024** Bundesmittel 5 1 5 13 29 161 316 159 213 357 Eigenkapitalerhöhung* 33 71 30 67 65 Mittel DB AG 2 22 18 35 62 112 83 225 200 Die kaufmännischen Systeme der DB AG orientieren sich an einer projektbasierten Struktur und nicht an Innovationsfeldern. Eine Aufteilung im Sinne der Fragestellung liegt nicht vor. Alle Mittel werden vollständig in die Realisierung der Projekte eingebracht. Die Vergabequote im Rahmen des Wettbewerbsverfahrens lag 2024 bei 82,5 Prozent für die Projekte der Digitalisierung der Schiene.</code>                                                                                                                                                                                                                                                                                                                   | <code>0.0</code> |
  | <code>Wie viele Mittel haben Bund und DB AG insgesamt seit Beginn der Förderung von ETCS im Jahr 2015 (https://bmdv.bund.de/SharedDocs/DE/ Artikel/E/digitalisierung-der-schiene.html) für die Digitalisierung der Schiene aufgewendet (bitte einzeln pro Jahr auflisten)? a) Wie verteilen sich diese Mittel gemäß den Innovationsfeldern des Programms „Digitale Schiene Deutschland“. Hierzu zählen ETCS, DSTW, iLBS [integriertes Leitund Bediensystem], ATO [Automatic Train Operation] GoA [Grade of Automation] 2, CTMS [Capacity & Traffic Management System], ATO GoA 4, ADI [Advanced Digital Infrastructure] und FRMCS [Future Railway Mobile Communication System] (https://digitale-schiene-deutschland.de/Vision-und-Ziel bild)? b) Wie viele der Mittel fließen in Projektrealisierungen und wie viele anteilig in Vergaben in Wettbewerbsverfahren? * Eigenkapitalerhöhung im Zusammenhang mit dem Klimaschutzprogramm 2030. ** Vorläufige Daten für das Jahr 2024.</code> | <code>Die Aufwendungen von Bund und DB AG für die Digitalisierung der Schiene können der folgenden Tabelle entnommen werden. [in Mio. Euro] 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024** Bundesmittel 5 1 5 13 29 161 316 159 213 357 Eigenkapitalerhöhung* 33 71 30 67 65 Mittel DB AG 2 22 18 35 62 112 83 225 200 Die kaufmännischen Systeme der DB AG orientieren sich an einer projektbasierten Struktur und nicht an Innovationsfeldern. Eine Aufteilung im Sinne der Fragestellung liegt nicht vor. Alle Mittel werden vollständig in die Realisierung der Projekte eingebracht. Die Vergabequote im Rahmen des Wettbewerbsverfahrens lag 2024 bei 82,5 Prozent für die Projekte der Digitalisierung der Schiene.</code>                                                                                                                                                                                                                                                                                                                   | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 2
- `learning_rate`: 0.0001
- `warmup_steps`: 0.1
- `gradient_accumulation_steps`: 2
- `load_best_model_at_end`: True
- `dataloader_pin_memory`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 0.0001
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 2
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: False
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch   | Step     | Training Loss | Validation Loss | val_average_precision |
|:-------:|:--------:|:-------------:|:---------------:|:---------------------:|
| 0.0409  | 24       | 0.8353        | -               | -                     |
| 0.0818  | 48       | 0.8642        | -               | -                     |
| 0.1227  | 72       | 0.7839        | -               | -                     |
| 0.1635  | 96       | 0.6142        | -               | -                     |
| 0.2010  | 118      | -             | 0.5275          | 0.8629                |
| 0.2044  | 120      | 0.5521        | -               | -                     |
| 0.2453  | 144      | 0.5574        | -               | -                     |
| 0.2862  | 168      | 0.5538        | -               | -                     |
| 0.3271  | 192      | 0.4788        | -               | -                     |
| 0.3680  | 216      | 0.4601        | -               | -                     |
| 0.4020  | 236      | -             | 0.4158          | 0.8959                |
| 0.4089  | 240      | 0.4933        | -               | -                     |
| 0.4497  | 264      | 0.5073        | -               | -                     |
| 0.4906  | 288      | 0.4500        | -               | -                     |
| 0.5315  | 312      | 0.5021        | -               | -                     |
| 0.5724  | 336      | 0.4650        | -               | -                     |
| 0.6031  | 354      | -             | 0.3988          | 0.9059                |
| 0.6133  | 360      | 0.4468        | -               | -                     |
| 0.6542  | 384      | 0.4691        | -               | -                     |
| 0.6951  | 408      | 0.4555        | -               | -                     |
| 0.7359  | 432      | 0.4598        | -               | -                     |
| 0.7768  | 456      | 0.3905        | -               | -                     |
| 0.8041  | 472      | -             | 0.3765          | 0.9205                |
| 0.8177  | 480      | 0.4190        | -               | -                     |
| 0.8586  | 504      | 0.4472        | -               | -                     |
| 0.8995  | 528      | 0.3708        | -               | -                     |
| 0.9404  | 552      | 0.3872        | -               | -                     |
| 0.9813  | 576      | 0.4567        | -               | -                     |
| 1.0051  | 590      | -             | 0.3549          | 0.9249                |
| 1.0221  | 600      | 0.3926        | -               | -                     |
| 1.0630  | 624      | 0.3624        | -               | -                     |
| 1.1039  | 648      | 0.4074        | -               | -                     |
| 1.1448  | 672      | 0.3382        | -               | -                     |
| 1.1857  | 696      | 0.4127        | -               | -                     |
| 1.2061  | 708      | -             | 0.3658          | 0.9240                |
| 1.2266  | 720      | 0.3617        | -               | -                     |
| 1.2675  | 744      | 0.4175        | -               | -                     |
| 1.3083  | 768      | 0.3301        | -               | -                     |
| 1.3492  | 792      | 0.4298        | -               | -                     |
| 1.3901  | 816      | 0.3188        | -               | -                     |
| 1.4072  | 826      | -             | 0.3421          | 0.9335                |
| 1.4310  | 840      | 0.4068        | -               | -                     |
| 1.4719  | 864      | 0.4111        | -               | -                     |
| 1.5128  | 888      | 0.3827        | -               | -                     |
| 1.5537  | 912      | 0.3380        | -               | -                     |
| 1.5945  | 936      | 0.3921        | -               | -                     |
| 1.6082  | 944      | -             | 0.3318          | 0.9369                |
| 1.6354  | 960      | 0.3712        | -               | -                     |
| 1.6763  | 984      | 0.3678        | -               | -                     |
| 1.7172  | 1008     | 0.3794        | -               | -                     |
| 1.7581  | 1032     | 0.3355        | -               | -                     |
| 1.7990  | 1056     | 0.3474        | -               | -                     |
| 1.8092  | 1062     | -             | 0.3310          | 0.9382                |
| 1.8399  | 1080     | 0.2924        | -               | -                     |
| 1.8807  | 1104     | 0.3913        | -               | -                     |
| 1.9216  | 1128     | 0.3603        | -               | -                     |
| 1.9625  | 1152     | 0.3826        | -               | -                     |
| **2.0** | **1174** | **-**         | **0.3292**      | **0.9393**            |

* The bold row denotes the saved checkpoint.

### Training Time
- **Training**: 1.3 hours
- **Evaluation**: 28.7 minutes
- **Total**: 1.8 hours

### Framework Versions
- Python: 3.13.12
- Sentence Transformers: 6.0.1
- Transformers: 5.17.0
- PyTorch: 2.14.0
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->