# code-models-entity-tracking
Working repo for testing the effects of code training on entity tracking.


## Inference 
```
python main_autoregressive.py --model_name google/gemma-2b --test_file ../../entity-tracking-lms/data/boxes-dataset-v1/few_shot_boxes_nso_exp2_max3/test-subsample-states-t5.jsonl --output_path ../output/constrained_outlines/gemma_2b;
```

## Evaluation
```
python evaluate/expand_results.py --input_file  ../output/constrained_outlines/gemma_2b/predictions.jsonl --expanded_dataset ../../entity-tracking-lms/data/boxes-dataset-v1/few_shot_boxes_nso_exp2_max3/test-subsample-states-t5.jsonl; python evaluate/compute_metrics.py --model_output ../output/constrained_outlines/gemma_2b/predictions-expanded.jsonl --gold_data ../../entity-tracking-lms/data/boxes-dataset-v1/few_shot_boxes_nso_exp2_max3/test-subsample-states-t5.jsonl; 
```

