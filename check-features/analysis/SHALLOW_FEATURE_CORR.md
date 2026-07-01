# Predictions vs. shallow / semantic features (GPT-2-XL)

## Pereira — corr(prediction, sentence length)   n=627
| predictor | r with length |
|---|---|
| EMB (act) | +0.277 |
| EMB (sum) | +0.564 |
| SFL+RFF | +0.758 |

## Tuckute — corr(prediction, feature)   rated subset n=1000
| feature | group | EMB (act) | EMB (sum) | SFL+RFF |
|---|---|---|---|---|
| rating_imageability_mean | semantic | -0.499 | -0.411 | -0.246 |
| rating_physical_mean | semantic | -0.444 | -0.367 | -0.109 |
| rating_places_mean | semantic | -0.372 | -0.314 | -0.068 |
| rating_others_thoughts_mean | semantic | -0.084 | -0.060 | -0.177 |
| rating_valence_mean | semantic | -0.144 | -0.143 | -0.097 |
| rating_arousal_mean | semantic | -0.089 | -0.062 | -0.177 |
| rating_sense_mean | semantic | -0.294 | -0.277 | -0.429 |
| avg_word_length | form | +0.214 | +0.204 | +0.433 |
| avg_word_freq_zipf | form | -0.120 | -0.123 | -0.438 |
| rating_frequency_mean | form | -0.254 | -0.239 | -0.515 |
| rating_conversational_mean | form | -0.198 | -0.198 | -0.413 |
| rating_gram_mean | form | -0.329 | -0.307 | -0.430 |
| log-prob-pcfg_mean | form | -0.168 | -0.174 | -0.347 |
| dep_length_mean | form | +0.029 | +0.033 | +0.095 |

## UCL-N400 — corr(prediction, covariate)   n=1505
| covariate | EMB (act) | EMB (sum) | SFL+RFF |
|---|---|---|---|
| surprisal | -0.305 | -0.252 | -0.588 |
| log-frequency | +0.331 | +0.278 | +0.493 |
| length | -0.251 | -0.228 | -0.314 |
