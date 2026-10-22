# RouteB R4 AVAXUSDT hierarchy comparison — run 02 outcome

Status: failed under the registered compute safety stop; run-02 timestamps are consumed and cannot be retried.

- Fold: `7959f22680bb5d6b5c0f76428f3b39201353e2238565e9a395847100f70e62a0`
- Read scope: `[2020-11-24T00:01:00Z, 2021-12-07T00:01:00Z)`
- Source snapshot: `8bb41785cc3137e1055be0363f6ce121e89ab6d2f36b6c1b96e24af89f31f193`
- Test and Holdout payloads decoded: `false`
- Completed work before failure: E3 completed all three seeds and wrote all three checkpoints, path files, and run reports.
- Failure: E3b/seed-31 validation inference attempted to allocate 4.34 GiB in a single vectorized forecast call and JAX returned `RESOURCE_EXHAUSTED` on the 8 GiB GPU.
- Failure report SHA-256: `a512cccd3fca8ea87c4486abfe6490bf7d46ffafbd69fe581814903824eefea7`
- Research receipt: `failed`.

The forecast implementation now bounds GPU memory by splitting the ordered seed list into small chunks and concatenating results on the host. The model, path order, number of candidate seeds, and scored-path selection rule are unchanged. Run 03 must use a third disjoint Fold.
