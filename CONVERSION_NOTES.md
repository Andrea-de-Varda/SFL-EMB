# Conversion notes (per-dataset wrangling recipes)

Working notes for converting each dataset onto the `src/` pipeline. Captures the
non-obvious wrangling so the conversions stay faithful. Delete before submission
(or fold the key points into the README methods).

Shared conventions: sentences are word-lists INCLUDING the sentence-initial word
(dropped via `extract_words(drop_first=True)` when aligning to behavioral/target
rows); SFL = [surprisal, log-freq(nat), length]; RFF n_components = D (hidden size),
median-heuristic gamma, leakage-safe (fit on train rows/words only); outer CV =
10-fold sequential except where noted; per-fold StandardScaler + fit_intercept=False.

## Behavioral — DONE
- **Provo** (`probe_Provo.py`): word-level, no spill. First word missing from CSV -> recovered from stimuli file. fp=IA_FIRST_RUN_DWELL_TIME (A1), ff=IA_FIRST_FIXATION_DURATION, tf=IA_DWELL_TIME (A3). Uses `behavioral.run_word_dataset`. VALIDATED (SFL/EMB within 0.01 of published).
- **Meco** (`probe_meco.py`): word-level. Sentences from behavioral data (en texts). fp=firstrun.dur, ff=firstfix.dur, tf=dur. Exclude subids ee_09/ee_22/ru_8; unfixated->0.
- **ZuCo** (`probe_zuco.py`): word-level. Shipped CSV = 349 NR sentences only, participant-averaged. fp=gd, ff=ff, tf=tt; drop word_idx 0; mask ~isnan(gd).

## Behavioral — TODO
- **Maze** (`probe_maze_natstor_nospill.py` -> rewrite): word-level, NO spill. Read `natstor_maze.rds` via pyreadr (`maze[None]`). Boyce exclusions: native English; subject attentive (mean(correct=="yes")>0.8); per-word `error` flag (scan: successor word error=1 unless both current & previous correct; non-successor error=1 if current wrong); filter = type!=practice & 100<rt<5000 & is_attentive & word_num!=0 & error==0. groupby [sent_n,word_num,word,unique] mean rt (raw ms, no transform). Sentences from `corpora/maze_sentences.txt`; sent_n remapped from FILE ORDER (`{" ".join(line):idx}`). Target = per-word mean rt; many non-initial words excluded -> need row_mask selecting words present in aggregated data keyed (sent_n,word_num). N≈9779. Fix bug: original baseline_large used s_GPT (use s_large). Results -> results/maze_results.csv.
- **Brown** (`probe_Brown_NatStor_spill.py` Brown half -> rewrite): SPR, spill i,i-1,i-2 on BOTH SFL(9) and EMB(3D). Behavioral `bnc-brown.csv` col `psychometric` (pre-aggregated RT, raw). Sentences `corpora/bnc_brown.txt`. word_pos from create_word_pos over sent[1:] (first non-initial word=word_pos 2). Targets from word_pos>=4 (need i,i-1,i-2 all non-initial). length=len(raw word). Fix bug: original `*_all` third lag duplicated current word (use real i-2). Use `behavioral.run_spill_dataset` with sent_ids. N = sum(max(0,L-3)).
- **NatStor_SPR** (`probe_NatStor_spill_realigned.py` -> rewrite): SPR, spill i,i-1,i-2 on SFL & EMB. `datasets/natstor.tsv` (item,zone,word,gmeanItemRT). Split stories into sentences on end punct `[.!?]["')\]]*$`. Target=gmeanItemRT. length AFTER punct-strip (differs from Brown; standardize on raw len). Drop non-finite rows. target from 4th word onward. Use `run_spill_dataset`. Results -> results/natstor_spr_results.csv.

## Brain — TODO
Word→unit mapping + pooling via `src/rff.pool_words_to_units` (sentence) / `aggregate_to_tr` (TR).

- **UCL_N400** (`probe_UCL_N400_matched.py` -> rewrite): word-level EEG, no spill. Read `UCL_reading_all.csv`; target col `N400` (precomputed: 12 centro-parietal electrodes 300-500ms, participant-avg, artefact-excluded upstream). Exclusions: dropna(s_GPT2_small) [drops sentence-initial] then keep words with NEITHER "." NOR "," (two-char test, NOT the broad _PUNCT), then drop NaN N400. Sentences from stimuli (sorted by sent_nr), drop_first. SFL cols [s_*, logfreq, length] from CSV OR recompute. EMB=mean pooling, last layer. DROP emb_sum (not one of my 5). Map old base/act/all/sfl_rff/all_matched -> sfl/emb/all/sfl_rff/sfl_rff_emb. n_components=D. 10-fold sequential. N≈1505. Results -> results/results_UCL_N400_matched.csv.

- **Pereira** (`probe_Pereira_passages.py` -> rewrite): sentence-level fMRI, PASSAGE-GROUPED GroupKFold(10) [KEEP - decided]. `pereira_1/pereira_averaged_with_passage.csv`: y=`EffectSize` (already ROI/participant-averaged), groups=`passage_group`, 627 sentences (Exp2+3, pre-filtered in CSV). `SFL.csv` per-word cols [s_*, freq, len, sentnum]. EMB: drop first token, MEAN pool per sentence (also emb_sum but I drop it), last layer. SFL base: MEAN pool [s,len,freq] by sentnum. SFL_RFF: per-word SFL -> RFF(D) -> SUM pool per sentence. wsi=sentnum. My 5 conditions: sfl(mean-pool 3d), emb(mean-pool D), all, sfl_rff(RFF sum-pool), sfl_rff_emb=hstack(sfl_rff, emb). Keep legacy p_base_emb/p_all_base/p_all_emb for plotting. Results -> pereira_1/results/results.csv. NOTE: RFF+pooling needs a custom build_fn (fit RFF on train-sentence words, sum-pool per sentence) since pooling is per-fold — use fit.run_folds with a build_fn over word_sfl+wsi.

- **Tuckute/control** (`control_fmri/control_encoding.py`+`probe_control_matched.py` -> rewrite): sentence-level fMRI, 10-fold SEQUENTIAL (groups=None). `control_fmri/data/brain-lang-data_participant_20230728.csv`, roi=="lang_LH_netw", groupby sentence mean of `response_target` = y (~2000 sents). `SFL.csv` [s_*, freq, len, sentnum]. EMB mean-pool per sentence, last layer, drop first token. Sentences order = groupby-sorted (written to data/sentences.txt). Same 5-condition mapping as Pereira but sequential folds. Results -> control_fmri/results/control_results.csv.

- **Wehbe/HP** (`fMRI/fMRI_analysis_HP_averaged.py`+`probe_HP_matched.py` -> rewrite): TR fMRI, 10-fold sequential over 1211 TRs. Subjects data_subject_{F,H,I,J,K,L,M,N}.npy (1211,27905); HP_subj_roi_inds.npy dict roi[part]["all"]. y = mean over 8 parts of part[:, roi["all"]].mean(axis=1) -> (1211,) using the "all" language network (7 AAL regions available but canonical uses "all"). Word→TR: words_id[i]=where(time_words[i]>time)[0][-1] (last TR onset before word); time_words trimmed [20:5176]. Per-run trim [20:-15] over runs 1..4 (1351->1211) AFTER aggregate_to_tr. EMB mean-pool per TR (drop first 20 words), last layer; SFL mean-pool per TR. Lags 1..4 -> 4*D. SFL_RFF: per-word SFL->RFF(D)->SUM per TR->4 lags. n_components=D. Results -> fMRI/results/all_results.csv (use ABSOLUTE dataset-scoped path to avoid collision).

- **NatStor_fMRI** (`natstor_fmri/fit_encoding_natstor.py`+`probe_natstorfmri_matched.py` -> rewrite): TR fMRI, **LEAVE-ONE-STORY-OUT (9 folds), NOT 10-fold** [KEEP LOSO - appropriate CV, document like Pereira]. y from `response/d_natstor` pickle (5 lang ROIs, top-10% voxels upstream, participant-avg, trimmed [8:-8]). 9 stories ids 1-7,9,10. Word→TR: drop first CSV row; time=arange(0,n*2,2); wid=where(end>time)[0][-1] (uses END time). EMB pickles `embeddings/{prefix}_{story}` dict[last_layer][1:]; SFL pickles `sfl_{prefix}_{story}` (already first-word-dropped, 3-dim). Lagging WITHIN story before concat. EMB mean-pool/TR, SFL_RFF sum-pool/TR, lags 1..4 -> 4*D (all_matched 8*D). Folds = per-story test-index ranges into concatenated y. Results -> natstor_fmri/results/all_results.csv (dataset-scoped path).

## Bugs fixed in conversion (vs originals)
- Maze `baseline_large` used s_GPT -> use s_large (uniform per-model build avoids).
- Brown `*_all` third lag duplicated current word -> real i-2 (uniform spill avoids).
- Meco/ET ffix/totfix significance reused wrong baseline handle -> uniform naming avoids.
- probe_ET N400/P600 sig loop appended stale p_base_emb -> not an issue in new pipeline.
- Permutation test: +1 identity-permutation correction (was omitted).
- fMRI all_results.csv relative-path collision -> dataset-scoped absolute paths.
- fMRI alpha grid deviations (0.01..1e5 etc) -> unified widened grid.
