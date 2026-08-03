# Experiments

## text/google_gemma3_causallm/001_qat_transfer

run_id params: model, task, mode, seed, data_spec, train_spec, qv_source, alpha, quantizer, eval_spec (mirrors `RUN_ID_PARAMS` in `run_task.py`)

expected final artifact: `evaluations/text/google_gemma3_causallm/001_qat_transfer/<run_id path>/eval_results.json`

| model | task | mode | seed | data spec | train spec | QV source | alpha | quantizer | eval spec | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---:|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-125849 | behemoth | 0,2 | failed |  | preflight |  | 2026-08-03 21:39 |  | no run launched; raw FP32 delta failed exact BF16 reconstruction |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-125849 | behemoth | 4,5 | failed |  | preflight |  | 2026-08-03 21:39 |  | no run launched; raw FP32 delta failed exact BF16 reconstruction |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-125849 | behemoth | 6,7 | failed |  | preflight |  | 2026-08-03 21:39 |  | no run launched; raw FP32 delta failed exact BF16 reconstruction |
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-213921 | behemoth | 0,2 | failed |  | preflight |  | 2026-08-03 21:52 |  | no run launched; 12 embedding cancellations cannot be represented by FP32 delta |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-213921 | behemoth | 4,5 | failed |  | preflight |  | 2026-08-03 21:52 |  | no run launched; 12 embedding cancellations cannot be represented by FP32 delta |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-213921 | behemoth | 6,7 | failed |  | preflight |  | 2026-08-03 21:52 |  | no run launched; 12 embedding cancellations cannot be represented by FP32 delta |
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-215248 | behemoth | 0,2 | failed |  | rendezvous preflight |  | 2026-08-03 22:01 |  | no run launched; standalone rendezvous chose unresolvable public hostname |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-215248 | behemoth | 4,5 | failed |  | rendezvous preflight |  | 2026-08-03 22:01 |  | no run launched; standalone rendezvous chose unresolvable public hostname |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-215248 | behemoth | 6,7 | failed |  | rendezvous preflight |  | 2026-08-03 22:01 |  | no run launched; standalone rendezvous chose unresolvable public hostname |
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-220134 | behemoth | 0,2 | failed | 2026-08-03 22:05 | conversion preflight |  | 2026-08-03 22:10 |  | smoke trained/resumed; elastic child omitted llama.cpp module path; no full run launched |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-220134 | behemoth | 4,5 | failed |  | conversion preflight |  | 2026-08-03 22:10 |  | no full run launched; replacement follows smoke finding |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-220134 | behemoth | 6,7 | failed |  | conversion preflight |  | 2026-08-03 22:10 |  | no full run launched; replacement follows smoke finding |
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-221250 | behemoth | 0,2 | failed | 2026-08-03 22:18 | tokenizer-copy preflight |  | 2026-08-03 22:19 |  | smoke trained; double-applied tokenizer regex fix failed before conversion |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-221250 | behemoth | 4,5 | failed |  | tokenizer-copy preflight |  | 2026-08-03 22:19 |  | no full run launched; replacement follows smoke finding |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-221250 | behemoth | 6,7 | failed |  | tokenizer-copy preflight |  | 2026-08-03 22:19 |  | no full run launched; replacement follows smoke finding |
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-222037 | behemoth | 0,2 | failed | 2026-08-03 22:25 | conversion preflight |  | 2026-08-03 22:27 | 1m57s | smoke trained/resumed; converter read persisted regex-fix flag and tried to patch fixed tokenizer again |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-222037 | behemoth | 4,5 | failed |  | conversion preflight |  | 2026-08-03 22:27 |  | no full run launched; replacement follows smoke finding |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-222037 | behemoth | 6,7 | failed |  | conversion preflight |  | 2026-08-03 22:27 |  | no full run launched; replacement follows smoke finding |
| gemma-3-1b-it | gsm8k | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-223300 | behemoth | 0,2 | todo |  |  |  |  |  | serialized tokenizer marked regex-fix-consumed; smoke-gated |
| gemma-3-1b-it | samsum | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-223300 | behemoth | 4,5 | todo |  |  |  |  |  | serialized tokenizer marked regex-fix-consumed; smoke-gated |
| gemma-3-1b-it | e2e_nlg | full | 2038 | equal6449_v1 | emnlp2025_fullft_v1 | gemma-3-1b-it-qat-q4_0 | 1.0 | llamacpp-b9637-q4_0 | gemma_gen_v1 | 20260803-223300 | behemoth | 6,7 | todo |  |  |  |  |  | serialized tokenizer marked regex-fix-consumed; smoke-gated |

## 998_rebuttal/003_lambda_sensitivity/001_signed_bert
run_id params: model, split, alpha, receiver, donors (mirrors RUN_ID_PARAMS in run_row.py)
expected final artifact: evaluations/.../<run_id path>/complete.json

| model | split | alpha | receiver | donors | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---:|---|---|---|---|---:|---|---|---|---|---|---|---|
| bert-large | val | 0.0 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | Banking77 | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | Emotion | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | IMDB | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | MTOPDomain | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | MTOPIntent | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | MassiveIntent | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | MassiveScenario | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | ToxicConversations | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | 0.0 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | Banking77 | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | Emotion | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | IMDB | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | MTOPDomain | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | MTOPIntent | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | MassiveIntent | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | MassiveScenario | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | ToxicConversations | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.05 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | Banking77 | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | Emotion | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | IMDB | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | MTOPDomain | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | MTOPIntent | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | MassiveIntent | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | MassiveScenario | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | ToxicConversations | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.1 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | Banking77 | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | Emotion | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | IMDB | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | MTOPDomain | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | MTOPIntent | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | MassiveIntent | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | MassiveScenario | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | ToxicConversations | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.15 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | Banking77 | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | Emotion | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | IMDB | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | MTOPDomain | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | MTOPIntent | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | MassiveIntent | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | MassiveScenario | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | ToxicConversations | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.2 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | Banking77 | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | Emotion | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | IMDB | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | MTOPDomain | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | MTOPIntent | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | MassiveIntent | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | MassiveScenario | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | ToxicConversations | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.25 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | Banking77 | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | Emotion | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | IMDB | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | MTOPDomain | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | MTOPIntent | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | MassiveIntent | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | MassiveScenario | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | ToxicConversations | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.3 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | Banking77 | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | Emotion | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | IMDB | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | MTOPDomain | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | MTOPIntent | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | MassiveIntent | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | MassiveScenario | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | ToxicConversations | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.35 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | Banking77 | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | Emotion | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | IMDB | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | MTOPDomain | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | MTOPIntent | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | MassiveIntent | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | MassiveScenario | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | ToxicConversations | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.4 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | Banking77 | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | Emotion | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | IMDB | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | MTOPDomain | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | MTOPIntent | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | MassiveIntent | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | MassiveScenario | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | ToxicConversations | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.45 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | Banking77 | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | Emotion | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | IMDB | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | MTOPDomain | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | MTOPIntent | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | MassiveIntent | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | MassiveScenario | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | ToxicConversations | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.5 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | Banking77 | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | Emotion | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | IMDB | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | MTOPDomain | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | MTOPIntent | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | MassiveIntent | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | MassiveScenario | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | ToxicConversations | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -0.75 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | Banking77 | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | Emotion | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | IMDB | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | MTOPDomain | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | MTOPIntent | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | MassiveIntent | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | MassiveScenario | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | ToxicConversations | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.0 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | Banking77 | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | Emotion | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | IMDB | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | MTOPDomain | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | MTOPIntent | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | MassiveIntent | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | MassiveScenario | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | ToxicConversations | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.25 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | Banking77 | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | Emotion | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | IMDB | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | MTOPDomain | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | MTOPIntent | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | MassiveIntent | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | MassiveScenario | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | ToxicConversations | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.5 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | Banking77 | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | Emotion | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | IMDB | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | MTOPDomain | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | MTOPIntent | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | MassiveIntent | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | MassiveScenario | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | ToxicConversations | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -1.75 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | Banking77 | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | Emotion | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | IMDB | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | MTOPDomain | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | MTOPIntent | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | MassiveIntent | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | MassiveScenario | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | ToxicConversations | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | val | -2.0 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | AmazonCounterfactual | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | AmazonReviewsClassification | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | Banking77 | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | Emotion | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | IMDB | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | MTOPDomain | all | 20260802-152022 | behemoth | 0 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | MTOPIntent | all | 20260802-152022 | behemoth | 2 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | MassiveIntent | all | 20260802-152022 | behemoth | 4 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | MassiveScenario | all | 20260802-152022 | behemoth | 5 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | ToxicConversations | all | 20260802-152022 | behemoth | 6 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |
| bert-large | test | -1.0 | TweetSentimentExtraction | all | 20260802-152022 | behemoth | 7 | done | 2026-08-02 15:54 | 11/11 |  | 2026-08-02 19:02 | 3h08m | initial wave complete |

## 998_rebuttal/005_qv_alignment — ViT-B/16 Euclidean pilot

Normative design and ordered producer/analyzer run identities:
`code/experiments/998_rebuttal/005_qv_alignment/RESEARCH_NOTE.md`.

Golden artifacts:

- geometry: `evaluations/998_rebuttal/005_qv_alignment/euclidean_alignment/<producer_run_id_path>/euclidean_alignment.json`
- analysis: `evaluations/998_rebuttal/005_qv_alignment/euclidean_alignment/<producer_run_id_path>/analysis/<analyzer_run_id_path>/euclidean_statistics.json`
- figures: `plots/998_rebuttal/005_qv_alignment/<script_stem>/<mirrored_run_id_path>/`

Implementation and targeted verification are complete. In the first real wave,
the geometry producer succeeded, while the analyzer failed at startup because
Hydra parsed an unescaped `=` in an override path. The analyzer-only retry wave
succeeded, and all three real figures were generated as PDF+PNG.

| stage | model | dependency | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---|---|---:|---|---|---|---|---|---|---|
| `compute_euclidean_alignment` | `vit_base_patch16_224.orig_in21k` | canonical final FP/QAT checkpoints for fixed 22-task set | 20260802-162236 | rig-4090 | 0 | done | 08-02 16:28 | vectors 22/22; artifact written |  | 08-02 16:29 | 1m08s | global `H=I` cosine over quantized `nn.Linear.weight` subspace; no partial resume; W&B off |
| `analyze_euclidean_alignment` | `vit_base_patch16_224.orig_in21k` | validated `euclidean_alignment.json` plus existing 22x22 full-QV test outcome JSON | 20260802-190820 | rig-4090 | 0 | done | 08-02 19:10 | comparisons 4/4; artifact written |  | 08-02 19:10 | 2.5s | `reviewer_3hfp_v1`; inference on 462 cross-task cells; 10,000 QAP permutations |
| Euclidean visualization suite | `vit_base_patch16_224.orig_in21k` | validated `euclidean_statistics.json` | 20260802-190820 | rig-4090 | 0 | done | 08-02 19:10 | 3/3 figures; 6/6 files verified |  | 08-02 19:15 |  | real heatmap, association, and influence figures rendered as PDF+PNG |

### Wave 20260802-162236

The producer's canonical flat identity is 415 bytes, so its scripts/logs folder
uses the approved SHA-256 alias
`producer_run_id_sha256=da1865269b54e2e48c978428ed1700f22eda648c0102432f3fcda2d3b4122ec9`.
Its scientific nested evaluation identity is unchanged. Both stages are
CPU-only; `gpu0` is the standard lane identity and CUDA is not invoked.

| stage | run identity | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---|---:|---|---|---|---|---|---|---|
| `compute_euclidean_alignment` | `producer_run_id_sha256=da1865269b54e2e48c978428ed1700f22eda648c0102432f3fcda2d3b4122ec9` | 20260802-162236 | rig-4090 | 0 | done | 08-02 16:28 | vectors 22/22; artifact written |  | 08-02 16:29 | 1m08s | first real run; no partial resume; golden `euclidean_alignment.json` present and nonempty |
| `analyze_euclidean_alignment` | `ptq_bits=3,ptq_granularity=channel,outcome_protocol=full_qv,outcome_split=test,unit_alpha=1.0,analysis_spec=reviewer_3hfp_v1,n_permutations=10000,permutation_seed=2038` | 20260802-162236 | rig-4090 | 0 | failed | 08-02 16:29 | startup failure before analysis |  | 08-02 16:29 | <1s | Hydra override parse error: unescaped `=` in path; golden `euclidean_statistics.json` absent; no real figures generated |

### Wave 20260802-190820

Analyzer-only retry on the same approved rig-4090 CPU lane after fixing Hydra
string quoting. The successful geometry artifact from wave `20260802-162236`
was reused without recomputation.

| stage | run identity | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---|---:|---|---|---|---|---|---|---|
| `analyze_euclidean_alignment` | `ptq_bits=3,ptq_granularity=channel,outcome_protocol=full_qv,outcome_split=test,unit_alpha=1.0,analysis_spec=reviewer_3hfp_v1,n_permutations=10000,permutation_seed=2038` | 20260802-190820 | rig-4090 | 0 | done | 08-02 19:10 | comparisons 4/4; artifact written |  | 08-02 19:10 | 2.5s | golden `euclidean_statistics.json` valid (3,220,749 bytes); all 6 real figure files present and nonempty |

## 998_rebuttal/006_alignment_alpha_response — Level A alpha response

Normative design: Step 7 of
`code/experiments/998_rebuttal/005_qv_alignment/RESEARCH_NOTE.md`.

Analyzer run-id params, in order: `model_name`, `curve_split`,
`curve_baseline`, `curve_grid`, `analysis_spec`, `n_permutations`,
`permutation_seed`. The golden artifact is
`evaluations/998_rebuttal/006_alignment_alpha_response/<run_id_path>/alpha_response_statistics.json`.
W&B is disabled; this read-only join creates no checkpoints and has no partial
resume. The CPU-only analyzer completed in wave `20260802-211910`; gpu 0 is
the standard lane identity and CUDA was not invoked. The render-only
visualization suite subsequently completed plainly on rig-4090: 3/3 figures,
6/6 nonempty files (PDF plus 300-dpi PNG), structurally and visually verified.

| model_name | curve_split | curve_baseline | curve_grid | analysis_spec | n_permutations | permutation_seed | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---|---|---:|---:|---|---|---:|---|---|---|---|---|---|---|
| `vit_base_patch16_224.orig_in21k` | val | fp_ptq | shared | `reviewer_3hfp_alpha_v1` | 10000 | 2038 | 20260802-211910 | rig-4090 | 0 | done | 08-02 21:22 | statistics and artifact 4/4; artifact written |  | 08-02 21:22 | 2.9s | golden JSON 12,331,488 bytes; source hashes verified; no traceback; W&B off |


## vision/ilharco_timm_supervised/009_qat_transfer_awq

run_id params: model, src, tgt, optim, qat, awq, qv, split (mirrors `RUN_ID_PARAMS` in `qv_transfer_awq.py`)

Golden artifact: `evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq/<run_id path>/eval_results.json`

| model | src | tgt | optim | qat | awq | qv | split | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---|---|---|---|---|---|---|---:|---|---|---|---|---|---|---|
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | Cars-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:25 | 4/4 |  | 08-02 16:27 | 2m18s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | DTD-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:28 | 4/4 |  | 08-02 16:29 | 1m19s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | EuroSAT-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:34 | 4/4 |  | 08-02 16:35 | 1m25s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | GTSRB-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:48 | 4/4 |  | 08-02 16:51 | 2m53s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | MNIST-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:04 | 4/4 |  | 08-02 17:07 | 2m29s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | RESISC45-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:15 | 4/4 |  | 08-02 17:17 | 1m58s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | SUN397-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:22 | 4/4 |  | 08-02 17:30 | 7m47s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | SVHN-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:30 | 4/4 |  | 08-02 17:35 | 4m47s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | CIFAR10-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:20 | 4/4 |  | 08-02 16:22 | 2m27s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | CIFAR100-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:22 | 4/4 |  | 08-02 16:25 | 2m27s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | STL10-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:19 | 4/4 |  | 08-02 17:21 | 2m12s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | Food101-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:43 | 4/4 |  | 08-02 16:48 | 4m46s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | Flowers102-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:40 | 4/4 |  | 08-02 16:43 | 2m31s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | FER2013-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:35 | 4/4 |  | 08-02 16:37 | 2m04s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | PCAM-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:09 | 4/4 |  | 08-02 17:15 | 5m49s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | OxfordIIITPet-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:07 | 4/4 |  | 08-02 17:09 | 1m36s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | RenderedSST2-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:18 | 4/4 |  | 08-02 17:19 | 1m19s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | EMNIST-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:29 | 4/4 |  | 08-02 16:33 | 4m03s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | FashionMNIST-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:38 | 4/4 |  | 08-02 16:40 | 2m29s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | KMNIST-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:01 | 4/4 |  | 08-02 17:04 | 2m29s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | TinyImageNet-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:35 | 4/4 |  | 08-02 17:37 | 2m30s |  |
| vit_base_patch16_224_orig_in21k | ImageNet-seed2038 | ImageNet-seed2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead | b3-gchannel-shead-n4-grid20-clip1 | a1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:53 | 4/4 |  | 08-02 17:01 | 8m21s |  |

## vision/ilharco_timm_supervised/010_awq_transfer

W&B disabled. The materialized checkpoint is weights-only, named `classifier_epoch_1.pt`, retained permanently, and does not support resume.

### materialize_awq_checkpoint

run_id params: model, donor, seed, optim, awq (mirrors `RUN_ID_PARAMS` in `materialize_awq_checkpoint.py`)

Golden artifact: `$CHECKPOINT_BASE_PATH/vision/ilharco_timm_supervised/awq_transfer/<run_id path>/classifier_epoch_1.pt`

| model | donor | seed | optim | awq | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---:|---|---|---|---|---:|---|---|---|---|---|---|---|
| vit_base_patch16_224_orig_in21k | ImageNet | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 16:17 | 3/3 |  | 08-02 16:19 | 2m25s | no resume; calibration on ImageNet train |

### qv_transfer_awqv

run_id params: model, src, tgt, sseed, tseed, optim, awq, alpha, split (mirrors `RUN_ID_PARAMS` in `qv_transfer_awqv.py`)

Golden artifact: `evaluations/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv/<run_id path>/eval_results.json`

| model | src | tgt | sseed | tseed | optim | awq | alpha | split | wave | rig | gpu | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---:|---:|---|---|---:|---|---|---|---:|---|---|---|---|---|---|---|
| vit_base_patch16_224_orig_in21k | ImageNet | Cars | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:39 | 2/2 |  | 08-02 17:39 | 29s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | DTD | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:39 | 2/2 |  | 08-02 17:39 | 18s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | EuroSAT | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:41 | 2/2 |  | 08-02 17:41 | 15s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | GTSRB | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:44 | 2/2 |  | 08-02 17:44 | 37s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | MNIST | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:49 | 2/2 |  | 08-02 17:49 | 30s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | RESISC45 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:51 | 2/2 |  | 08-02 17:52 | 23s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | SUN397 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:53 | 2/2 |  | 08-02 17:54 | 1m48s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | SVHN | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:54 | 2/2 |  | 08-02 17:56 | 1m10s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | CIFAR10 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:37 | 2/2 |  | 08-02 17:38 | 33s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | CIFAR100 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:38 | 2/2 |  | 08-02 17:39 | 31s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | STL10 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:52 | 2/2 |  | 08-02 17:52 | 26s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | Food101 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:42 | 2/2 |  | 08-02 17:44 | 1m10s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | Flowers102 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:42 | 2/2 |  | 08-02 17:42 | 30s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | FER2013 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:41 | 2/2 |  | 08-02 17:41 | 24s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | PCAM | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:50 | 2/2 |  | 08-02 17:51 | 1m38s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | OxfordIIITPet | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:49 | 2/2 |  | 08-02 17:50 | 17s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | RenderedSST2 | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:52 | 2/2 |  | 08-02 17:52 | 13s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | EMNIST | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:40 | 2/2 |  | 08-02 17:40 | 54s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | FashionMNIST | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:41 | 2/2 |  | 08-02 17:42 | 30s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | KMNIST | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:48 | 2/2 |  | 08-02 17:49 | 32s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | TinyImageNet | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:56 | 2/2 |  | 08-02 17:56 | 31s |  |
| vit_base_patch16_224_orig_in21k | ImageNet | ImageNet | 2038 | 2038 | lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128 | b3-gchannel-shead-n4-grid20-clip1 | 1.0 | test | 20260802-160930 | rig-3090-ti | 0 | done | 08-02 17:44 | 2/2 |  | 08-02 17:48 | 3m38s |  |
| bert-large | val | -2.25 | AmazonCounterfactual | AmazonReviewsClassification | 20260802-152022 | behemoth | 0 | done | 2026-08-02 19:41 | 1/1 |  | 2026-08-02 19:42 | 1m | boundary extension complete |
| bert-large | val | -2.5 | AmazonCounterfactual | AmazonReviewsClassification | 20260802-152022 | behemoth | 2 | done | 2026-08-02 19:41 | 1/1 |  | 2026-08-02 19:42 | 1m | boundary extension complete |
| bert-large | val | -2.75 | AmazonCounterfactual | AmazonReviewsClassification | 20260802-152022 | behemoth | 4 | done | 2026-08-02 19:41 | 1/1 |  | 2026-08-02 19:42 | 1m | boundary extension complete |
| bert-large | val | -3.0 | AmazonCounterfactual | AmazonReviewsClassification | 20260802-152022 | behemoth | 5 | done | 2026-08-02 19:41 | 1/1 |  | 2026-08-02 19:42 | 1m | boundary extension complete |
| bert-large | test | -1.75 | AmazonCounterfactual | AmazonReviewsClassification | 20260802-152022 | behemoth | 0 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | AmazonCounterfactual | Emotion | 20260802-152022 | behemoth | 2 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.5 | AmazonCounterfactual | MTOPIntent | 20260802-152022 | behemoth | 4 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | AmazonReviewsClassification | AmazonCounterfactual | 20260802-152022 | behemoth | 5 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.45 | AmazonReviewsClassification | IMDB | 20260802-152022 | behemoth | 6 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.3 | AmazonReviewsClassification | MTOPIntent | 20260802-152022 | behemoth | 7 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.25 | AmazonReviewsClassification | MassiveScenario | 20260802-152022 | behemoth | 0 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.15 | AmazonReviewsClassification | TweetSentimentExtraction | 20260802-152022 | behemoth | 2 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.2 | Banking77 | IMDB | 20260802-152022 | behemoth | 4 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.1 | Banking77 | MTOPIntent | 20260802-152022 | behemoth | 5 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | Emotion | IMDB | 20260802-152022 | behemoth | 6 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.15 | IMDB | MTOPIntent | 20260802-152022 | behemoth | 7 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | MTOPDomain | Banking77 | 20260802-152022 | behemoth | 0 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.2 | MTOPDomain | IMDB | 20260802-152022 | behemoth | 2 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.2 | MTOPDomain | MTOPIntent | 20260802-152022 | behemoth | 4 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.15 | MTOPDomain | MassiveScenario | 20260802-152022 | behemoth | 5 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.15 | MTOPDomain | TweetSentimentExtraction | 20260802-152022 | behemoth | 6 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.15 | MTOPIntent | IMDB | 20260802-152022 | behemoth | 7 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | MassiveIntent | Banking77 | 20260802-152022 | behemoth | 0 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.2 | MassiveIntent | IMDB | 20260802-152022 | behemoth | 2 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | MassiveIntent | MTOPIntent | 20260802-152022 | behemoth | 4 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | MassiveIntent | MassiveScenario | 20260802-152022 | behemoth | 5 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | MassiveIntent | TweetSentimentExtraction | 20260802-152022 | behemoth | 6 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.25 | MassiveScenario | IMDB | 20260802-152022 | behemoth | 7 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.2 | MassiveScenario | MTOPIntent | 20260802-152022 | behemoth | 0 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | MassiveScenario | TweetSentimentExtraction | 20260802-152022 | behemoth | 2 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | ToxicConversations | IMDB | 20260802-152022 | behemoth | 4 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.05 | ToxicConversations | MTOPIntent | 20260802-152022 | behemoth | 5 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.1 | ToxicConversations | MassiveScenario | 20260802-152022 | behemoth | 6 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.75 | TweetSentimentExtraction | IMDB | 20260802-152022 | behemoth | 7 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |
| bert-large | test | -0.1 | TweetSentimentExtraction | MTOPIntent | 20260802-152022 | behemoth | 0 | done | 2026-08-02 20:03 | 1/1 |  | 2026-08-02 20:20 | 17m | selected test complete |

## Reviewer 3HFP — ImageNet-only strong-PTQ alpha sweep

Wave `20260802-212527` is one adaptive dispatch decision. Stage 1 evaluates
the paper alpha grid `[0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05,
1.20, 1.35, 1.50]` on validation for 21 ImageNet-to-other-dataset receivers,
independently under AWQ and GPTQ (42 jobs, 462 result cells). Each job owns a
receiver row so all 11 alphas reuse frozen receiver calibration batches. Stage
2 will select by `val_accuracy_fp_head_awq` or `val_accuracy_fp_head_gptq` and
evaluate only the 42 frozen choices on test. ImageNet self-transfer is excluded.

The user explicitly authorized behemoth GPUs `0,2,4,5,6,7` for this wave.
Rig-4090 was excluded after preflight because `nvidia-smi` could not communicate
with its driver. Detailed placement and immutable job lists are in
`scripts/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/wave_20260802-212527/manifest.json`.

| lane | wave | rig | gpu | jobs | cells | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|
| behemoth_gpu0 | 20260802-212527 | behemoth | 0 | 6 | 66 | done | 08-02 22:03 | 66/66 |  | 08-02 23:06 | 1h04m | user-authorized GPU; all golden artifacts present |
| behemoth_gpu2 | 20260802-212527 | behemoth | 2 | 7 | 77 | done | 08-02 22:03 | 77/77 |  | 08-02 23:14 | 1h11m | user-authorized GPU; all golden artifacts present |
| behemoth_gpu4 | 20260802-212527 | behemoth | 4 | 7 | 77 | done | 08-02 22:03 | 77/77 |  | 08-02 23:08 | 1h06m | user-authorized GPU; all golden artifacts present |
| behemoth_gpu5 | 20260802-212527 | behemoth | 5 | 7 | 77 | done | 08-02 22:03 | 77/77 |  | 08-02 23:19 | 1h17m | user-authorized GPU; all golden artifacts present |
| behemoth_gpu6 | 20260802-212527 | behemoth | 6 | 7 | 77 | done | 08-02 22:03 | 77/77 |  | 08-02 23:23 | 1h20m | user-authorized GPU; all golden artifacts present |
| behemoth_gpu7 | 20260802-212527 | behemoth | 7 | 7 | 77 | done | 08-02 22:03 | 77/77 |  | 08-02 23:13 | 1h11m | user-authorized GPU; all golden artifacts present |
| rig-3090-ti_gpu0 | 20260802-212527 | rig-3090-ti | 0 | 1 | 11 | done | 08-02 21:45 | 11/11 |  | 08-02 22:05 | 20m | AWQ/Food101; all golden artifacts present |

Stage 2 freezes one validation-best alpha per method/receiver using the stated
metric and exact-tie rule `smallest_alpha`, then evaluates those 42 choices on
test. The frozen choices and candidate scores are in `selected_alphas.json`.

| selected-test lane | wave | rig | gpu | jobs | status | started | progress | eta | ended | elapsed | notes |
|---|---|---|---:|---:|---|---|---|---|---|---|---|
| behemoth_gpu0 | 20260802-212527 | behemoth | 0 | 6 | done | 08-03 09:22 | 6/6 |  | 08-03 09:31 | 9m35s | user-authorized GPU; all golden artifacts present |
| behemoth_gpu2 | 20260802-212527 | behemoth | 2 | 7 | done | 08-03 09:22 | 7/7 |  | 08-03 09:32 | 10m32s | user-authorized GPU; all golden artifacts present |
| behemoth_gpu4 | 20260802-212527 | behemoth | 4 | 7 | done | 08-03 09:22 | 7/7 |  | 08-03 09:33 | 11m20s | user-authorized GPU; all golden artifacts present |
| behemoth_gpu5 | 20260802-212527 | behemoth | 5 | 7 | done | 08-03 09:22 | 7/7 |  | 08-03 09:33 | 11m52s | user-authorized GPU; all golden artifacts present |
| behemoth_gpu6 | 20260802-212527 | behemoth | 6 | 7 | done | 08-03 09:22 | 7/7 |  | 08-03 09:34 | 12m43s | user-authorized GPU; all golden artifacts present |
| behemoth_gpu7 | 20260802-212527 | behemoth | 7 | 7 | done | 08-03 09:22 | 7/7 |  | 08-03 09:33 | 11m33s | user-authorized GPU; all golden artifacts present |
| rig-3090-ti_gpu0 | 20260802-212527 | rig-3090-ti | 0 | 1 | done | 08-03 09:20 | 1/1 |  | 08-03 09:24 | 4m43s | AWQ/Food101 alpha=0.3; golden artifact present |
