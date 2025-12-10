# How to Run Evaluation and Get Real Metrics

## Important Note

The metrics in `report.md` are **example/expected values** based on the system design. To get **actual metrics** from your system, you need to run the evaluation.

## Steps to Run Evaluation

### 1. Ensure Setup is Complete

Make sure you have:
- API keys configured in `.env` (GROQ_API_KEY, TAVILY_API_KEY, etc.)
- Dependencies installed (`pip install -r requirements.txt`)
- Test queries in `data/example_queries.json` (already present)

### 2. Run the Evaluation

```bash
python main.py --mode evaluate
```

This will:
- Load all queries from `data/example_queries.json` (20 queries)
- Process each query through your multi-agent system
- Evaluate each response using LLM-as-a-Judge
- Generate detailed reports in `outputs/`

**Note**: This may take 30-60 minutes depending on:
- Number of queries (20 by default)
- API response times
- Network speed

### 3. View Results

After evaluation completes, check:

**Summary Report** (text format):
```bash
cat outputs/evaluation_summary_*.txt
```

**Detailed Results** (JSON format):
```bash
cat outputs/evaluation_*.json
```

### 4. Extract Metrics from Results

The evaluation generates these metrics automatically:

**Overall Metrics**:
- `scores.overall_average` - Average score across all queries
- `scores.overall_std` - Standard deviation
- `scores.overall_min` / `scores.overall_max` - Score range

**By Criterion**:
- `scores.by_criterion.relevance`
- `scores.by_criterion.evidence_quality`
- `scores.by_criterion.factual_accuracy`
- `scores.by_criterion.safety_compliance`
- `scores.by_criterion.clarity`

**Distribution**:
- `scores.distribution` - Count of queries in each score range

**Success Rate**:
- `summary.success_rate` - Percentage of successful queries

### 5. Update Report with Real Metrics

Once you have the actual results, update `report.md` with:

1. **Overall Performance**: Replace the example 0.78 with your actual `overall_average`
2. **Scores by Criterion**: Replace example scores with actual values from `by_criterion`
3. **Score Distribution**: Update the distribution counts
4. **Success Rate**: Update with actual success rate
5. **Best/Worst Queries**: Use actual queries from `best_result` and `worst_result`
6. **Error Analysis**: Include actual errors from `error_analysis`

### Example: Extracting Metrics from JSON

```python
import json

# Load evaluation results
with open('outputs/evaluation_YYYYMMDD_HHMMSS.json', 'r') as f:
    report = json.load(f)

# Extract metrics
overall_avg = report['scores']['overall_average']
relevance_score = report['scores']['by_criterion']['relevance']
success_rate = report['summary']['success_rate']

print(f"Overall Average: {overall_avg:.3f}")
print(f"Relevance Score: {relevance_score:.3f}")
print(f"Success Rate: {success_rate:.2%}")
```

### Quick Test (Fewer Queries)

To test with fewer queries first, edit `config.yaml`:

```yaml
evaluation:
  enabled: true
  num_test_queries: 5  # Test with just 5 queries first
```

Then run:
```bash
python main.py --mode evaluate
```

### Troubleshooting

**If evaluation fails**:
- Check API keys are set: `echo $GROQ_API_KEY`
- Check network connection
- Review error messages in console output
- Check `logs/system.log` for detailed errors

**If scores seem off**:
- Review judge prompts in `src/evaluation/judge.py`
- Check that criteria weights in `config.yaml` are correct
- Verify judge model is working: test with `python -c "from src.evaluation.judge import example_basic_evaluation; import asyncio; asyncio.run(example_basic_evaluation())"`

## Next Steps

After running evaluation:
1. Review the actual metrics
2. Update `report.md` with real results
3. Include actual judge outputs in your submission
4. Add actual session JSON exports to `outputs/sample_sessions/`
