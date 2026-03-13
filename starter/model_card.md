# Model Card

## Model Details

This project uses a `RandomForestClassifier` trained on the UCI Census Income
dataset included in [starter/data/census.csv](/home/andreas/tmp-src/nd0821-c3-starter-code/starter/data/census.csv).
The trained model, encoder, and label binarizer are stored in
[starter/model](/home/andreas/tmp-src/nd0821-c3-starter-code/starter/model).

## Intended Use

The model predicts whether a person's annual salary is `<=50K` or `>50K` based
on demographic and employment features. It is intended for the course project
only and should not be used to make real employment, credit, housing, or legal
decisions.

## Training Data

The training data is the cleaned version of the Adult Census Income dataset in
[starter/data/census_clean.csv](/home/andreas/tmp-src/nd0821-c3-starter-code/starter/data/census_clean.csv).
The cleaning step strips the inconsistent whitespace from the raw CSV column
names and string values but leaves the feature semantics unchanged.

## Evaluation Data

The model is evaluated on a stratified 20% holdout split created from the same
cleaned dataset with `random_state=42`.

## Metrics

Holdout-set performance from [starter/model/metrics.json](/home/andreas/tmp-src/nd0821-c3-starter-code/starter/model/metrics.json):

- Precision: `0.7570`
- Recall: `0.6378`
- F1: `0.6923`

Per-slice metrics for each categorical feature are written to
[starter/model/slice_output.txt](/home/andreas/tmp-src/nd0821-c3-starter-code/starter/model/slice_output.txt).

## Ethical Considerations

This dataset includes sensitive demographic attributes such as sex, race, and
native country. Predictions may reflect historical bias in the data and should
be treated as a classroom exercise rather than a fair or policy-safe system.

## Caveats and Recommendations

The model uses a single train/test split and minimal preprocessing. It should
be treated as a baseline implementation. Before any production use, the project
would need stronger validation, threshold analysis, feature review, and bias
assessment across sensitive slices.
