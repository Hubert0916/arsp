# ASRP source/topic analysis

以 150 個 public binary sources 中可取得的 142 個 source、每個 source 第一個 subject、每個 subject 40 題 held-out items 進行分析；共 5,680 題。採用和 submission 相同的 `adaptive_4_warm`，比較 ASRP 與同一題的 smoothed term。這是 public proxy，不是官方 hidden draw。

## Overall

|budget|ASRP Brier|smoothed Brier|difference|ASRP bias|
|---:|---:|---:|---:|---:|
|0|0.2330|0.2450|-0.0120|-0.0214|
|1|0.2155|0.2234|-0.0079|-0.0199|
|3|0.2063|0.2108|-0.0045|-0.0006|
|7|0.1961|0.1989|-0.0028|+0.0008|
|15|0.1899|0.1899|-0.0000|+0.0017|
|31|0.1801|0.1849|-0.0048|-0.0029|

Positive difference means ASRP is worse than the smoothed term. ASRP is slightly better overall at every budget, but the advantage at budget 31 is only about 0.0048 Brier.

## Sources where ASRP loses most at budget 31

|source|n|ASRP|smoothed|difference|bias|
|---|---:|---:|---:|---:|---:|
|ood_prediction|40|0.3248|0.2499|+0.0748|+0.085|
|edu_circuit_hw|40|0.3156|0.2510|+0.0646|-0.083|
|faithcot|40|0.2832|0.2392|+0.0440|-0.038|
|reliancescope|40|0.2664|0.2234|+0.0430|-0.067|
|kormedmcqa|40|0.1609|0.1204|+0.0405|-0.227|
|clasheval|40|0.2360|0.2007|+0.0353|+0.122|
|helm_afr|40|0.2749|0.2400|+0.0349|+0.042|
|llm_uncertainty_bench|40|0.2845|0.2531|+0.0314|+0.127|
|helm_thaiexam|40|0.2679|0.2366|+0.0313|-0.177|
|medhelm_medcalc_bench|40|0.1916|0.1687|+0.0229|+0.224|
|icpc2_code_selector|40|0.2123|0.1903|+0.0220|-0.063|
|sgrades|40|0.2197|0.1994|+0.0202|-0.031|
|real_webagents|40|0.2180|0.1983|+0.0197|-0.178|
|legal_rag_bench|40|0.2070|0.1875|+0.0195|-0.062|
|pxplore|40|0.2401|0.2206|+0.0195|-0.089|

The most consistent source-level weaknesses are `ood_prediction`, `edu_circuit_hw`, `faithcot`, `reliancescope`, `helm_afr`, and `helm_thaiexam`. `ood_prediction` and `edu_circuit_hw` become worse after labels accumulate, which points to evidence/kernel over-update rather than a simple cold-start problem.

## Explicit item topics

Only 35.2% of analysed items contain a usable topic-like `item_features` field; the rest have no topic metadata and are reported as `source=<name>`. Topic rows below have at least 10 items.

|topic|n|ASRP|smoothed|difference|bias|
|---|---:|---:|---:|---:|---:|
|task=ravel_task|15|0.3570|0.2499|+0.1071|+0.328|
|dataset=records|11|0.3780|0.2935|+0.0846|-0.392|
|category=vs|12|0.3030|0.2508|+0.0522|-0.061|
|prompt_origin=transcribed_evaluation_report_not_wire_request|40|0.2664|0.2234|+0.0430|-0.067|
|dataset=math500|11|0.2698|0.2320|+0.0378|-0.141|
|task=pricetag_task|12|0.2811|0.2503|+0.0308|-0.247|
|difficulty=challenging|19|0.2305|0.1998|+0.0308|+0.240|
|dataset=scientsbank_3way|14|0.2349|0.2044|+0.0305|+0.003|
|task=truthfulqa|15|0.2476|0.2294|+0.0182|-0.148|
|category=foodchainswebs|12|0.1876|0.1697|+0.0178|-0.215|
|category=general-vqa|20|0.2426|0.2276|+0.0150|+0.085|
|task=time|11|0.1840|0.1704|+0.0135|-0.208|
|category=replace_rel|10|0.1815|0.1691|+0.0123|+0.100|
|task=input|24|0.2547|0.2431|+0.0116|-0.012|
|category=injection|12|0.2659|0.2570|+0.0090|+0.114|

The clearest topic-level regressions are `task=ravel_task`, `dataset=records`, `category=vs`, `task=pricetag_task`, and `dataset=math500`. These groups are small (11–15 items except the source fallback), so they require shrinkage and should not receive a hard per-topic correction.

## Modeling implication

1. Use topic metadata only when it is present and has enough public support; shrink topic residuals toward the global/source residual.
2. Add a guard for sources whose Brier worsens as budget increases. For these sources, reduce kernel/evidence weight after budget 7 instead of assuming more labels always help.
3. For blank `item_features`, derive a coarse, frozen topic from the benchmark metadata or a small text classifier. Do not create hundreds of one-off content clusters.
4. Validate any topic correction by source-held-out splits. The tables above are diagnostic and must not be used as an in-sample lookup table for hidden sources.
