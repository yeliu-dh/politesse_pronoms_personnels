# Analyse de l‘expression de la politesse et des pronoms personnels dans les descriptions de hôtes Airbnb

Ce projet explore la relation entre l'usage des pronoms personnels et les expressions de politesse dans les descriptions des hôtes sur Airbnb

## 🧠 Objectif

Étudier dans quelle mesure les pronoms de première, deuxième et troisième personne sont liés à des expressions de politesse dans les textes de la prrésentation de soi des hôtes.

## 🛠 Méthodologie

- Utilisation de **spaCy** pour l'extraction et la lemmatisation automatique des pronoms personnels dans les textes (anglais et français).
- Correction des erreurs fréquentes dans la détection.
- Conception d’un ensemble d’items décrivant la politesse selon trois dimensions :
  - **Émotionnelle**
  - **Comportementale**
  - **Cognitive**
- Application d’un modèle de classification zéro-shot[`MoritzLaurer/bge-m3-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0) pour détecter la présence de ces items dans les descriptions.
- Calcul d’un **score global de politesse** par moyenne pondérée.
- Régression linéaire OLS entre les fréquences des pronoms et le score de politesse.

### 🔧 Optimisation et sélection des items

Pour assurer la robustesse et la pertinence des items utilisés pour mesurer la politesse, plusieurs étapes de validation ont été réalisées :
1. **Calcul du Cronbach Alpha** pour évaluer la fiabilité interne de chaque dimension de politesse.
2. **Analyse factorielle exploratoire (EFA)** pour examiner la contribution de chaque item à sa dimension respective de politesse, en analysant la **corrélation** et le **poids** de chaque facteur.
3. Vérification de la **correlation de Pearson** entre les items et le score global de politesse calculé par moyenne pondérée.
4. Après l'optimisation, les items qui n'avaient pas un facteur de charge suffisamment élevé dans leur dimension (en particulier les items liés à la dimension **cognitive**) ont été **exclus** du modèle.


## 📈 Résultats

- Les trois types de pronoms sont significativement liés à la politesse.
- La **deuxième personne** a l’effet le plus marqué, ce qui confirme son rôle dans l’interaction interpersonnelle.
- Les pronoms sont davantage associés à la **politesse émotionnelle** qu’à la politesse comportementale.