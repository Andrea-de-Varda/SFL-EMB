# Predictions vs. shallow / semantic features (GPT-2-XL)

## Pereira — corr(prediction, sentence length)   n=627
| predictor | r with length |
|---|---|
| EMB | +0.249 |
| SFL+RFF | +0.753 |

## Tuckute — corr(prediction, feature)   rated subset n=1000
| feature | group | EMB | SFL+RFF |
|---|---|---|---|
| rating_imageability_mean | semantic | -0.532 | -0.230 |
| rating_physical_mean | semantic | -0.475 | -0.103 |
| rating_places_mean | semantic | -0.417 | -0.057 |
| rating_others_thoughts_mean | semantic | -0.074 | -0.153 |
| rating_valence_mean | semantic | -0.157 | -0.098 |
| rating_arousal_mean | semantic | -0.099 | -0.140 |
| rating_sense_mean | semantic | -0.287 | -0.415 |
| avg_word_length | form | +0.188 | +0.417 |
| avg_word_freq_zipf | form | -0.116 | -0.355 |
| rating_frequency_mean | form | -0.252 | -0.473 |
| rating_conversational_mean | form | -0.176 | -0.372 |
| rating_gram_mean | form | -0.325 | -0.429 |
| log-prob-pcfg_mean | form | -0.157 | -0.311 |
| dep_length_mean | form | +0.058 | +0.143 |

## UCL-N400 — corr(prediction, covariate)   n=1505
| covariate | EMB | SFL+RFF |
|---|---|---|
| surprisal | -0.319 | -0.702 |
| log-frequency | +0.368 | +0.565 |
| length | -0.281 | -0.337 |
