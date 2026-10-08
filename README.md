# NexusNode: GNN-Powered Tactical Drafting Engine
NexusNode is an automated MLOps pipeline and recommendation engine that uses Graph Neural Networks (GNN) to optimize team compositions in League of Legends. By analyzing high-Elo match data from multiple global regions, it maps champion synergies into a 64-dimensional vector space to provide real-time drafting intelligence.


![Application](./images/UserInterface.png)
---

## 💼 Business Impact & Problem Statement
### The Problem
In competitive MOBA games, the "Draft Phase" determines up to 60% of the match outcome. However, players and coaches often rely on static win-rate statistics or subjective "gut feelings." Traditional analytics fail to capture the latent synergies—the hidden mathematical relationships between champions that only emerge in high-level play.

### The Solution
NexusNode replaces static stats with a Dynamic Embedding Model. By treating champions as nodes, with edges for both teammates and opponents, the system learns which champions "belong together" **and** which ones beat each other, then recommends the pick that maximizes your team's predicted win probability against the actual enemy draft.

### Business Value
- Performance Optimization: Increases win probability by identifying non-obvious champion synergies.

- Scalability: Automated ETL pipelines handle global data (KR, NA, EUW, BR) without manual intervention.

- Personalization: Integrates Riot Games API to tailor recommendations to a specific player's "Comfort Pool" and mastery history.

---

## 🏗️ Technical Architecture
The project is structured as a modular MLOps Pipeline:

- Ingestion (collect_data.py): A weekly automated scraper of Ranked Solo/Duo games from Challenger players across 4 global regions.

- Transformation (eda.py): Deduplicates raw Riot API data, standardizes roles, and keeps only complete 5v5 drafts.

- Featurization (preprocess.py):

  - Builds per-champion stat features (StandardScaled) and a two-relation champion graph: **teammate (synergy)** edges and **opponent (counter)** edges.

  - Builds role eligibility and an observed lane-matchup table used as evidence in the UI.

- Learning (train_gnn.py): a relational GNN (one GCN per relation) produces role-aware champion embeddings, trained end-to-end on **match outcomes** through a draft-scoring head:

  - **Power**: each champion's individual strength
  - **Synergy**: a symmetric bilinear term for every pair of teammates
  - **Counters**: an antisymmetric bilinear term for **every ally-vs-enemy pair** (all 25), so enemy picks in any role shape the prediction
  - **Lane matchup**: an extra antisymmetric term for same-role opponents

  The antisymmetric enemy terms guarantee that swapping the two teams exactly flips the prediction. Hyperparameters are chosen with held-out validation and the model is compared against an allies-only ablation and a no-draft baseline (see `data/processed/model_metrics.json`).

- Deployment (app.py): A Streamlit draft assistant with champion portraits (Data Dragon), a live win-probability bar, and ranked recommendations explained by their synergy with each ally and their edge/weakness against each enemy.

---

## 🚀 Automation (CI/CD)
The project utilizes GitHub Actions to maintain model relevancy in the ever-shifting "League Meta":

Weekly Scrape & Retrain: Every Monday at 00:00 UTC, a headless runner:

- Scrapes ~1,000+ new high-Elo matches.

- Cleans and transforms the data.

- Retrains the draft model on the updated graph.

- Commits the refreshed data and `data/processed/nexus_model.pt` back to the repository.

---

## 🛠️ Installation & Usage
### Prerequisites
- Python 3.10+

- Riot Games API Key (Developer Portal)


1.  Setup

    Clone the repository:

    ```Bash
    git clone https://github.com/PTuccinardi/NexusNode.git
    cd NexusNode
    ```

2. Install dependencies:

    ```Bash
    pip install -r requirements.txt
    ```

3. Configure environment:
    Create a .env file in the root:

    ```
    RIOT_KEY=your_api_key_here
    ```

4. Run the application:

```Bash
streamlit run app.py
```
---

## 🧪 Model Performance
Held-out validation on 512 matches (2,050 training matches, 172 champions):

| Model | Log loss ↓ | AUC ↑ |
|---|---|---|
| **Full model (allies + enemies)** | **0.6879** | **0.542** |
| Allies-only ablation | 0.6881 | 0.538 |
| No draft information (side bias only) | 0.6904 | 0.500 |

- Embedding size: 8 dimensions + role embeddings, chosen by 3-fold cross-validation with early stopping
- Including enemy interactions improves over the allies-only model, and both beat the no-draft baseline
- Draft-only prediction is a low-signal problem (player skill and execution dominate outcomes at Challenger), so the model's job is to rank picks by a few points of win probability, not to call games. Accuracy improves as the weekly pipeline adds more matches.

![performance](./images/LinkedinPostImage2.png)
---

## 👨‍💻 Author

Paul Tuccinardi – Data Scientist & ML Engineer

M.S. Data Science | Pace University

Philosophy: "Good, Better, Best" — iterative improvement through data.

*Disclaimer: NexusNode isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends.*
