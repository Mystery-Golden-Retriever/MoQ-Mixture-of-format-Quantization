# Progress Update

I have resumed monitoring the `run_remaining_experiments.sh` script. Here's a summary of the current state:
1. The experiment script is currently executing Phase 2 (`Qwen2.5-1.5B_wt_e2e_6bit`). 
2. **Bug Fix**: I noticed that the evaluation steps in `run_remaining_experiments.sh` missed the `wikitext2` dataset argument for `--ppl-dataset`, which resulted in all `ppl` values being recorded as `null` in `summary.json`.
3. I have applied a fix to `run_remaining_experiments.sh` so that subsequent evaluations will correctly evaluate Perplexity on `wikitext2`.
4. I also wrote and started a background script (`scripts/reeval_ppl.sh`) to automatically re-evaluate the missing PPL for all the Phase 1 experiments that have already completed. 

The re-evaluation is running in the background and will update `results/summary.json` when finished. While we wait for the 6-bit end-to-end calibration and the PPL re-evaluations to finish, is there any specific task you would like me to prepare or focus on (e.g. updating the `Qwen_Evaluation_Results.md` report with Phase 1 results)?
