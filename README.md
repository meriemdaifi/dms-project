# DMS Project - Driver Monitoring System (ADAS)

## Conception et Simulation d'un Système ADAS basé sur le Traitement Vidéo

### Description

Ce projet implémente un système complet de surveillance du conducteur (DMS - Driver Monitoring System) dans le cadre des systèmes ADAS (Advanced Driver Assistance Systems). Le système détecte la fatigue et la distraction du conducteur à travers le traitement vidéo et génère des alertes appropriées.

### Architecture du Système

Le système est structuré en modules :

1. **Acquisition Vidéo** (`src/video_acquisition.py`) - Capture vidéo et génération d'images synthétiques
2. **Détection Visage/Yeux** (`src/face_eye_detection.py`) - CNN pour la détection et classification
3. **Estimation Pose de la Tête** (`src/head_pose.py`) - Algorithme PnP pour les angles yaw/pitch/roll
4. **Analyse Comportementale** (`src/behavioral_analysis.py`) - PERCLOS, fréquence de clignement, scores
5. **Logique ECU** (`src/ecu_decision.py`) - Prise de décision et génération d'alertes
6. **Scénarios de Test** (`src/scenarios.py`) - Attentif, fatigué, distrait, mixte
7. **Visualisation** (`src/visualization.py`) - Graphiques et tableaux de résultats

### Outils Utilisés

- **Python** avec PyTorch, OpenCV, NumPy, Pandas, Matplotlib
- **MATLAB/Simulink** pour la modélisation et simulation
- **LaTeX** pour le rapport de PFE

### Installation

```bash
pip install -r requirements.txt
```

### Utilisation

```bash
# Lancer la simulation complète
python main.py

# Lancer les tests unitaires
python -m unittest tests.test_dms -v
```

### Structure du Projet

```
dms-project/
├── src/                          # Code source Python
│   ├── video_acquisition.py      # Module d'acquisition vidéo
│   ├── face_eye_detection.py     # CNN détection visage/yeux
│   ├── head_pose.py              # Estimation pose de la tête
│   ├── behavioral_analysis.py    # Analyse comportementale
│   ├── ecu_decision.py           # Logique décisionnelle ECU
│   ├── scenarios.py              # Scénarios de test
│   └── visualization.py          # Visualisation des résultats
├── tests/
│   └── test_dms.py               # Tests unitaires (34 tests)
├── matlab/
│   ├── dms_simulation.m          # Simulation MATLAB
│   ├── dms_simulink_model.m      # Modèle Simulink
│   └── simulink_model/
│       └── ecu_stateflow.m       # Machine à états ECU
├── report/
│   └── rapport_pfe.tex           # Rapport LaTeX
├── main.py                       # Point d'entrée principal
└── requirements.txt              # Dépendances Python
```

### Scénarios de Validation

| Scénario | Durée | Description |
|----------|-------|-------------|
| Attentif | 120s | Conduite normale avec clignements réguliers |
| Fatigué | 120s | Fatigue progressive en 4 phases |
| Distrait | 120s | Distraction croissante (regards latéraux) |
| Mixte | 180s | Transitions entre différents états |

### Résultats

Le système détecte correctement :
- ✅ Conducteur attentif (fatigue < 0.3, distraction < 0.3)
- ✅ Fatigue progressive (PERCLOS croissant, alertes générées)
- ✅ Distraction (score de distraction > 0.3 lors des regards latéraux)
- ✅ Transitions entre états dans le scénario mixte