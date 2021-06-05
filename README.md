### Evaluation-for-Shadow-Detection
This is a Pytorch implementation of evaluation for shadow detection.
This code contains metrics: BER, P-Error, N-Error and ACC.

### Enviroment requirements
pytorch >= 1.0

### Run
```
python main.py --result_dir <results_folder> --gt_dir <gt_folder>
```
### Note
You should put all the detection results into <results_folder>, and the names of results should be the same as ground-truth.
