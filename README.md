# Biomed-Data-Mining

### Project Focus
Our project focuses on relation extraction in the biomedical domain. Specifically, we aim to identify and classify semantic relations between biomedical entities (e.g., drugs, diseases, chemicals, proteins) from text. Relation extraction is a fundamental task in biomedical text mining, supporting applications such as knowledge graph construction, drug repurposing, and literature-based discovery.

### Motivation
With the rapid growth of biomedical literature, manually curating entity relations has become infeasible. Automatic relation extraction can accelerate biomedical research by enabling large-scale knowledge discovery and improving access to structured information. This project is particularly interesting to us because it bridges natural language processing (NLP) with biomedical informatics, and allows us to explore how domain-specific language models (e.g., BioBERT, ClinicalBERT) can outperform general models like BERT and SciBERT.

### Project Plan
We will use several publicly available biomedical corpora with annotated relations, including DDI Extraction 2013 (SemEval), BioCreative V CDR, BioCreative VI ChemProt, BioRED, BioInfer, BeFree, and EU-ADR. For modeling, we plan to fine-tune and compare multiple pretrained transformer-based models: BERT, BioBERT, SciBERT, ClinicalBERT, and BioClinicalBERT.
Our pipeline will involve:
Preprocessing and normalizing datasets (entity tagging, sentence extraction).
Fine-tuning models for relation extraction.
Comparing models across datasets to examine domain transferability and robustness.

### Evaluation Plan
We will evaluate performance using standard classification metrics, including accuracy, micro-F1, macro-F1, and weighted-F1. Results will be compared across different pretrained models to determine which architecture performs best for biomedical relation extraction.
