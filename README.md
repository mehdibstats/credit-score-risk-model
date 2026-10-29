# Classification du score de crédit de clients bancaires

Projet de data science : prédire le score de crédit (**Good**, **Standard**, **Poor**) de clients à partir de leur historique bancaire mensuel.

## Données

- `train.csv` : 100 000 lignes, 12 500 clients, 8 mois par client (janvier à août), avec la cible `Credit_Score`.
- `test.csv` : 50 000 lignes, les mêmes clients sur 4 mois (septembre à décembre), sans la cible.
- Les fichiers ne sont pas inclus dans ce dépôt. Jeu de données : *Credit score classification* (Kaggle), à placer à la racine du projet.

Répartition des classes : Standard 53 %, Poor 29 %, Good 18 %.

## Méthode

1. **Nettoyage** (`clean()`) : conversion des colonnes numériques sales (suffixes `_`), suppression des valeurs aberrantes par rapport au profil de chaque client (revenu, EMI) ou par bornes (âge, taux d'intérêt, nombre de prêts), imputation par client (médiane ou valeur la plus fréquente), encodage multi-label des types de prêts.
2. **Variables de comportement** : moyenne par client, écart du mois à la moyenne du client, valeur du mois précédent (retards, utilisation du crédit, solde, demandes de crédit).
3. **Pipeline scikit-learn** : `ColumnTransformer` (imputation, standardisation, one-hot) puis modèle.
4. **Validation** : `StratifiedGroupKFold` par client (5 plis), pour qu'un même client n'apparaisse jamais à la fois dans le train et le test d'un pli.
5. **Modèles comparés** : régression logistique, RandomForest, HistGradientBoosting (réglé avec `RandomizedSearchCV`).

## Résultats (validation croisée par clients)

| Modèle | Macro-F1 |
|---|---|
| Régression logistique | 0,64 |
| RandomForest | 0,68 |
| HistGradientBoosting réglé | **0,69** |

Modèle final : accuracy 0,71, macro-F1 0,69, AUC 0,86 (référence « toujours Standard » : accuracy 0,53).

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Good | 0,58 | 0,69 | 0,63 |
| Poor | 0,73 | 0,66 | 0,69 |
| Standard | 0,75 | 0,75 | 0,75 |

## Variables les plus importantes (importance par permutation)

`Credit_Mix`, `Interest_Rate`, retard moyen de paiement (`Delay_from_due_date_mean`), `Num_Credit_Card`, `Outstanding_Debt`.

## Limites

- Le score change d'un mois à l'autre pour 58 % des clients, sans variable qui l'explique : cela plafonne la performance autour de 0,69. Régression logistique, RandomForest et boosting convergent vers 0,68 à 0,69.
- Good est la classe la moins fiable (précision 0,58).
- `Credit_Mix` pèse très lourd et ressemble à un indicateur déjà dérivé du score : à vérifier avant tout usage réel.
- Dans le test, le mois précédent est inconnu pour septembre, et chaque client n'a que 4 mois au lieu de 8.

## Utilisation

```bash
pip install -r requirements.txt
```

Placer `train.csv` et `test.csv` à la racine, puis exécuter `credit_risk_ml.py` (cellules `#%%`, compatibles VS Code / Jupyter). Le script produit `submission.csv` (colonnes `ID`, `Credit_Score`).
