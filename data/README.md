#### `data/README.md`
```markdown
# Interaction Datasets & User Simulation

To address the cold-start data scarcity paradigm inherent to low-resource settings, this framework relies on two key data pathways:

1. **International Benchmark Calibration:** Standardized ASSISTments 2009-2010 tracking records used to initialize structural parity for the baseline DKT configuration.
2. **Localized Target Simulation:** An algorithmic user simulation tracking matrix capturing 1,000 synthetic trajectory lines embedded with typical algebraic misconceptions (e.g., parsing Burmese statements evaluating "$x+5=12$" incorrectly as "$x=12+5$").

## Generating Simulated Data
Run data orchestration sequences to output files matching your schema configuration into this directory before starting long-term training cycles. Ensure output structures follow:
`[student_id, time_step, skill_index, raw_text_response, binary_correctness]`