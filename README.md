# NexusNode: GNN-Powered Tactical Drafting Engine
NexusNode is an automated MLOps pipeline and recommendation engine that uses Graph Neural Networks (GNN) to optimize team compositions in League of Legends. By analyzing high-Elo match data from multiple global regions, it maps champion synergies into a 64-dimensional vector space to provide real-time drafting intelligence.

---

## 💼 Business Impact & Problem Statement
### The Problem
In competitive MOBA games, the "Draft Phase" determines up to 60% of the match outcome. However, players and coaches often rely on static win-rate statistics or subjective "gut feelings." Traditional analytics fail to capture the latent synergies—the hidden mathematical relationships between champions that only emerge in high-level play.

### The Solution
NexusNode replaces static stats with a Dynamic Embedding Model. By treating champions as nodes and winning compositions as edges, the system learns which champions "belong together" in a team's tactical identity.

### Business Value
- Performance Optimization: Increases win probability by identifying non-obvious champion synergies.

- Scalability: Automated ETL pipelines handle global data (KR, NA, EUW, BR) without manual intervention.

- Personalization: Integrates Riot Games API to tailor recommendations to a specific player's "Comfort Pool" and mastery history.

---

## 🏗️ Technical Architecture
The project is structured as a modular MLOps Pipeline:

- Ingestion (collect_data.py): A weekly automated scraper targeting Challenger-level players across 4 global regions.

- Transformation (eda.py): Sanitizes raw Riot API data, standardizes roles, and removes statistical outliers.

- Featurization (preprocess.py):

  - Constructs a weighted graph of champion pairings.

  - Uses StandardScaler and PCA to reduce high-dimensional performance metrics into baseline features.

- Learning (train_gnn.py):

  - Implements a NexusGNN architecture using GCNConv layers.

  - Utilizes Negative Sampling with a 3.0 weight multiplier to maximize the "spread" between non-synergistic champions.

- Deployment (app.py): A Streamlit-based tactical dashboard that performs real-time Vector Synthesis to suggest optimal picks.

---

## 🚀 Automation (CI/CD)
The project utilizes GitHub Actions to maintain model relevancy in the ever-shifting "League Meta":

Weekly Scrape & Retrain: Every Monday at 00:00 UTC, a headless runner:

- Scrapes ~1,000+ new high-Elo matches.

- Cleans and transforms the data.

- Retrains the GNN on the updated graph.

- Commits the new champion_embeddings.pt weights back to the repository.

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

    Code snippet
    ```
    RIOT_KEY=your_api_key_here
    ```

4. Run the application:

```Bash
streamlit run app.py
```
---

## 🧪 Model Performance
Embedding Size: 64 Dimensions

Training Epochs: 1,000 (Weekly Automated Pipeline)

Optimization: Spread-optimization via weighted Negative Sampling to prevent "Vector Clumping."

---

## 👨‍💻 Author

Paul Tuccinardi – Data Scientist & ML Engineer

M.S. Data Science | Pace University

Philosophy: "Good, Better, Best" — iterative improvement through data.

*Disclaimer: NexusNode isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends.*