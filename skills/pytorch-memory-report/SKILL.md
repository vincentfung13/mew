---
name: pytorch-memory-report
description: Generate and interpret a standalone interactive HTML report from a PyTorch CUDA memory snapshot produced by torch.cuda.memory._dump_snapshot. Use when asked to visualize active memory over time, inspect the largest CUDA tensor allocations, attribute allocations to Python code paths, correlate allocation lifetimes with recorded external annotations, identify unusual memory behavior, or suggest evidence-based memory improvements from a .pkl trace.
---

# PyTorch Memory Report

Generate the report with the bundled standard-library-only script. Do not unpickle an untrusted snapshot: Python pickle files can execute arbitrary code.

## Workflow

1. Locate the `.pkl` snapshot created by `torch.cuda.memory._dump_snapshot`.
2. Run from the user's working directory:

   ```bash
   python3 <skill-dir>/scripts/render_memory_report.py SNAPSHOT.pkl --output memory-report.html
   ```

3. If the table would be too large, keep the default largest 2,000 lifetimes or adjust `--max-events`. Use `--max-events 0` to include all events.
4. Use `--device N` for a multi-GPU snapshot and `--min-size-mib N` to exclude small allocation lifetimes from the table. These filters do not change the total-memory graph.
5. Open the output locally and report its absolute path. Summarize the peak, largest attributed allocations, generated insights, and attribution limitations. Do not claim that an unknown or pre-capture allocation has a known origin.

## Analyze the findings

Use the generated `Insights and possible improvements` section as evidence, not as a final diagnosis. Add a short interpretation tailored to the user's code and workload:

1. Call out two to five findings with exact sizes, ratios, code paths, or annotation names.
2. Separate observations from hypotheses. Use language such as `the trace shows` for measured facts and `this may indicate` for inferred causes.
3. Rank improvements by likely peak-memory impact and implementation risk. Prefer changes aimed at allocations live at the peak over changes aimed only at cumulative allocation traffic.
4. Inspect the referenced source before recommending a code change. For attention, optimizer, mixed precision, or checkpointing suggestions, verify that the implementation actually uses the relevant pattern.
5. Recommend a comparison capture when a claim requires a baseline. One snapshot alone cannot establish that memory is abnormal or that an optimization helped.
6. State when attribution coverage is weak, capture began after initialization, annotations are absent, or the trace cannot distinguish expected caching from a leak.

Do not label high allocation traffic alone as a leak. Do not assume that end-active growth across a partial capture is unintended retention. Do not add independent allocation sizes to estimate savings unless their lifetimes overlap at the peak.

## Interpret the output

- Treat each allocation table row as one allocation lifetime. `Allocated` is that allocation's size, not total memory at that instant.
- Treat annotation-table `Allocated during event` as cumulative allocation traffic. It can exceed peak memory because blocks may be allocated, freed, and reused during one range.
- Treat the graph as reconstructed PyTorch active memory, not CUDA process memory or PyTorch reserved memory.
- Read `Before capture` as a lower-bound lifetime. Its real allocation time predates memory-history recording.
- Read `End of capture` as an allocation still live when the snapshot was dumped.
- Treat generated insights as heuristic leads. Confirm them against source and, where possible, a controlled comparison trace.
- Prefer the first non-library Python frame for `Event / code path`; expand a row to inspect the complete captured stack.
- Use external annotations when the snapshot contains them. NVTX ranges stored only in an Nsight report are unavailable to this script.
- Explain that stackless native/autograd allocations cannot be attributed reliably from this artifact alone.

## Capture guidance

For useful provenance, enable memory history before the operations being investigated and disable it after dumping:

```python
torch.cuda.memory._record_memory_history(enabled="all")
# profiled work
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
torch.cuda.memory._record_memory_history(enabled=None)
```

Keep warmup outside recording for a focused steady-state trace. Put initialization inside recording when parameter and optimizer-state origins are required.
