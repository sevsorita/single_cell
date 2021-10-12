# Progress report


## Predict on positive part of new data
The models for the differentially expressed subset of the data (ESR1_expression > |0.5|) that are trained on the old dataset are not able to predict the expression of the cells in the new dataset. Exactly the same results occur if we use a model trained on the positive part of the old dataset and predict on postive part on the baseline in the new dataset (ESR1_expression > 0.5). 

### Results on old dataset
![ESR1 results on old dataset](plots/ESR1_results.png)
### Results on new dataset
![ESR1 results on new dataset](plots/results_new_baseline_diff_exp.png)


## Training models on the new dataset
Currently running randomized search for tuning the model.