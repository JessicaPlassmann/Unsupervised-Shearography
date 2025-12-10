# Unsupervised Learning for Shearography-Based Defect Detection

This repository serves as a central collection point for data, code, and results related to our ongoing research on the **automated evaluation of shearographic measurement data**, with a particular focus on **unsupervised** and **weakly supervised learning** approaches.

We aim to explore and evaluate methods that reduce or eliminate the need for manual labeling in the context of defect detection using shearography.

## Repository Scope

The repository is structured to support a growing set of contributions in this research direction. It includes:

- A curated dataset of shearographic image data
- Representative image samples categorized by structure and defect presence
- Code used in our publications and experiments
- Scripts and models for unsupervised and weakly supervised learning tasks

New modules and datasets may be added in the future as this line of research evolves.

## Prepare the Dataset & Repository

1. Download the Shearographic Anomaly Detection Dataset (SADD) using [this DOI](https://doi.org/10.5281/zenodo.17631257): 10.5281/zenodo.17631257 (size: ~ 5GB)
2. Extract the content of the downloaded zip file into `./resources/data/dataset`, resulting in a folder structure like `./resources/data/dataset/SADD/images` etc.
3. [Set up your virtual environment](https://docs.python.org/3/library/venv.html) and install the required packages using `pip install -r requirements.txt` (be sure to install torch GPU support beforehand), see `./requirements.txt`
4. Source your virtual environment and run a script of your choice, e.g. `python ./scripts/unsupervised/train_eval_all.py` to train and evaluate the entire unsupervised pipeline


## Note to Reviewers

> This repository contains materials related to our **published work on unsupervised learning** ([10.1007/978-3-032-11442-6_22](https://doi.org/10.1007/978-3-032-11442-6_22)) and our **submitted manuscript on on weakly supervised learning** for shearographic data.  
>
> The full dataset and source code will be made public upon paper acceptance to ensure transparency and reproducibility.  
>
> If early access is required for the purpose of reviewing or further clarification, please don’t hesitate to reach out. My contact information is included in the submitted manuscript and I am happy to provide the necessary materials upon request.

---
Thank you very much for your time and consideration.

