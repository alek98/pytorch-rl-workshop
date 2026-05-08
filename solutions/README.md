# RL solutions

Ovo su referentna rešenja za zadatke iz
[`src/3. RL.ipynb`](../src/3.%20RL.ipynb). Ovaj folder je za predavača, dok su
starter fajlovi za studente u [`tasks/`](../tasks).

Fajlovi:

- `q_learning_solution.py` - rešena `Q_Agent` klasa za FrozenLake:
  - Q-learning trening
  - V-tabela
  - evaluacija agenta
  - dodatno: kazna za rupe i 6x6 FrozenLake helper
- `common.py` - pomoćne klase/funkcije koje dele DQN rešenja:
  - `RewardTracker`
  - `EvaluationResult`
  - `make_frozen_lake`
  - `evaluate_model`
- `problem_3_one_hot_dqn.py` - rešenje za one-hot wrapper i DQN trening.
- `problem_4_custom_mlp_dqn.py` - rešenje sa custom PyTorch `BaseFeaturesExtractor`.
- `problem_5_cartpole.py` - rešenje za CartPole pomoću DQN-a.

Pokretanje:

```bash
python solutions/q_learning_solution.py
python solutions/problem_3_one_hot_dqn.py
python solutions/problem_4_custom_mlp_dqn.py
python solutions/problem_5_cartpole.py
```

Instalacija zavisnosti:

```bash
python -m pip install -r requirements.txt
```

Napomena: `gamma` utiče na učenje jer kontroliše koliko agent vrednuje buduće
nagrade. Tokom evaluacije agent samo koristi naučenu politiku, pa se `gamma`
ne koristi za izbor akcije.
