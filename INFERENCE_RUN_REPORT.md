# E2E Inference Run Report
    
**Generated**: 2026-09-04 18:33:14
**Checkpoint**: `artifacts/trainings/unified_orchestrator_checkpoint.pt`
**Status**: ✅ PASSED

## Journey Validation
- **Health Check**: ✅ PASSED
- **Inference Request**: ✅ PASSED

### Response Metadata
```json
{
  "success": true,
  "action_probabilities": {
    "action_0": 0.5,
    "action_1": 0.3,
    "action_2": 0.2
  },
  "best_action": "action_0",
  "value_estimate": 0.75,
  "subproblems": [],
  "refinement_info": {
    "converged": false,
    "convergence_step": 2,
    "recursion_depth": 2
  },
  "performance_stats": {
    "inference_time_ms": 105.8489999268204,
    "device": "cpu"
  },
  "error": null
}
```
