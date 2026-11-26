# Recommended next steps

1. Proceed to design the person detection and person–PPE association prototype using the existing 320 checkpoint as an experimental baseline. Do not assert a violation from an absent detection; define an explicit unknown or ambiguous result.
2. Establish small labelled association scenarios covering one worker, overlapping workers, occlusion and uncertain evidence; test whether each PPE box is attributed to the correct person before integrating safety rules.
3. In parallel, resolve the most consequential annotation questions: agree class/visibility policy, then adjudicate 00269 and a small number of sparse-label crowd examples such as 00031 and 03295. Record decisions without changing the existing dataset; any later correction should create a separately versioned dataset and baseline.
4. Keep the test split held out until the model and association decisions are frozen. Continue to report detection and event-level results separately and record unsupported conditions.

No Step 4 implementation is included in this audit.
