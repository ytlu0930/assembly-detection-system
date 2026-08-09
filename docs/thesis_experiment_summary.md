# Thesis Experiment Summary

## 1. Research Problem

Free-form VLM inference can identify broad assembly error types reasonably well, but affected-part identity remains unreliable and can be wrong at high confidence. The experiments therefore progressed from prompt-level constraints to localization-guided candidate reduction and component-level verification. Results from different stages use different denominators and are descriptive, not evidence of statistical significance.

## 2. Stage 1 — Free-form VLM Baseline

The baseline evaluation reported Exact Set Match of 8.00%, Part F1 of 10.5263%, and false-confident identity rate at 0.80 of 88.00%. These values describe the baseline evaluation subset recorded in the research evolution table; they should not be compared as paired observations with the later three-case ROI experiment. The baseline established that broad error recognition did not imply reliable affected-part identity.

## 3. Stage 2 — Prompt / Candidate Constraint

Prompt and candidate constraints produced no clear improvement. Candidate Exact Match was 0%, At-least-one Recall was 33.33%, All-parts Recall was 0%, Part F1 was 28.57%, and false-confident identity at 0.80 was 66.67%. The candidate set remained 15/15 for the targeted cases, effectively the full inventory, so membership enforcement alone was too weak to identify the affected component.

## 4. Stage 3 — Localization-guided ROI Candidate Reduction

Offline ROI construction reduced the mean candidate set by 64.44%, retained confirmed Ground Truth in 3/3 cases, and retained `EYE_BALL` in 0/3 candidate sets. Candidate counts changed from 15 to 5, 5, and 6. This stage demonstrated useful search-space reduction, but the frozen localization proposals remained unverified and required manual review.

## 5. Stage 4 — ROI Direct Classification

On the three frozen cases, ROI Direct achieved 33.33% Exact Set Match, 33.33% At-least-one Recall, 33.33% All-parts Recall, 25.00% Part Precision, 25.00% Part Recall, and 25.00% Part F1. False-confident identity at 0.80 was 100%, and manual review was required in all cases. The affected-part verifier allowed zero wrong identities to escape.

## 6. Stage 5 — ROI Checklist Verification

The original model responses were schema-valid in 0/3 cases. All three used the same contract drift: `results` instead of `checks`, `CHECK` or `check_result` instead of `status`, string confidence, and no `evidence_summary`.

An experiment-only deterministic normalizer reparsed the immutable raw response content. It changed only known aliases and types, supplied a non-semantic marker when evidence text was absent, enforced exact candidate membership, and consulted no Ground Truth. The normalized analysis responses were schema-valid in 3/3 cases and semantically identical to the earlier label-free recovery. This is post-hoc compatibility handling and must not be described as original model schema compliance.

Normalized Checklist semantics achieved 33.33% Exact Set Match, 66.67% At-least-one Recall, 66.67% All-parts Recall, 42.86% Part Precision, 75.00% Part Recall, and 54.55% Part F1. False-confident identity at 0.80 was 57.14%; Unknown Rate was 33.33%; Manual Review Rate remained 100%.

## 7. Checklist Confusion Matrix

Across 16 candidate checks, the resolved-only matrix was TP=3, FP=4, TN=3, and FN=0. Six additional checks were UNCERTAIN. Resolved-only Accuracy was 60.00%, Precision 42.86%, Recall 100.00%, and F1 60.00%; the overall Uncertain Rate was 37.50%. UNCERTAIN was kept separate rather than being converted into PASS or FAIL.

## 8. Case Analysis

### missingpart-A01

Ground Truth was `PIN_RED_SHORT`. Direct and normalized Checklist both produced `PIN_RED_SHORT`, but the affected-part verifier remained in conflict because localization was unverified. `EYE_BALL` was not in the reduced candidate set.

### missingpart-B01

Ground Truth was `WHEEL_BLUE_SMALL`. Direct incorrectly predicted `PIN_RED_SHORT` at 0.85. Checklist left the wheel and several other candidates UNCERTAIN, producing no affected identity. This avoided another forced high-confidence wrong identity but did not recover the target component.

### wrongpart-B01

The confirmed swap set was `PIN_RED_SHORT|PIN_YELLOW`. Direct predicted two unrelated link parts. Checklist retained both swap identities but also flagged four false positives, so All-parts Recall passed while Exact Set Match failed. Paired ROI reasoning did not isolate a clean two-part swap.

## 9. System Safety

The Affected-Part Identity Verifier blocked every incorrect identity in this experiment: wrong identity escaped verifier was zero for both methods. The tradeoff was a 100% manual-review rate and one correctly identified case also being blocked. This behavior is intentionally fail closed while localization and identity confidence remain unverified.

## 10. Final Output Design Decision

Deterministic bbox, arrow, and text annotation was selected for the prototype output. It preserves source pixels, is reproducible, can suppress unverified boxes, and directly reflects verifier state. GPT Image is not claimed to be incapable; the prototype evaluation instead identified risks in structural consistency, upstream error propagation, latency/cost, and reproducibility that are unnecessary for this evidence-oriented output.

## 11. Limitations

- The ROI experiment contains only three cases.
- Confirmed extra-part samples were unavailable.
- Wrong-part and swap errors require more complex paired localization.
- The original Checklist response contract was unstable at 0/3 schema validity.
- Normalization is post-hoc, experiment-only compatibility handling.
- Manual review remained 100%.
- Denominators vary across the baseline, prompt, and ROI stages.

## 12. Final Research Conclusion

Localization-guided candidate reduction and Checklist decomposition show a promising descriptive signal: the search space narrowed, confirmed targets remained available, semantic recall/F1 increased relative to ROI Direct, and false-confidence decreased. However, three cases, original contract instability, swap over-flagging, and universal manual review prevent a production-method recommendation. The supported conclusion is to retain deterministic fail-closed annotation, harden the experimental contract offline, and keep Phase 2B blocked pending broader validation.
