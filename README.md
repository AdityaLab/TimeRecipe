# TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness

## Publication

Implementation of the paper "TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness."

Authors: Zhiyuan Zhao, Juntong Ni, Haoxin Liu, Shangqing Xu, Wei Jin, B.Aditya Prakash

Paper + Appendix:  https://arxiv.org/abs/2506.06482

## Usage

### Training TimeRecipe

Please follow the training scripts provided in [TimeRecipeResults](https://github.com/AdityaLab/TimeRecipeResults).

To train a single setup

```
python -u run.py --seed 2021 --task_name long_term_forecast --use_norm "True" --use_decomp "True" --fusion "temporal" --emb_type "token" --ff_type "mlp" --${Other Args}$
```

To train a batch of setup

```
bash scripts/ecl_96_m/2021.sh 
```

or a customized batch of experiments aross datasets

```
bash run_2021.sh
```

### TimeRecipe Results

All raw and processes results can be found at [TimeReciperesults](https://github.com/AdityaLab/TimeRecipeResults).

### Results Post Processing

1. `./notebook/error_rank.ipynb`: Convert the raw forecasting results over different random seeds to ranked results with averaged error and std.
2. `./notebook/read_res_m.ipynb`: Filter and combine the top 30 ranked results (`top_k=30`) from different datasets to a single csv file.
3. `./notebook/cor_ana_m.ipynb`: Perform statistic testing for the correlation analysis, using the combined csv file (Paper Table 3).
4. `./notebook/lightgbm_m.ipynb`: Perform the training-free model selection using a LightGBM model and pre-trained results (Paper Table 2).
5. `./notebook/count_surpass.ipynb`: Count the number of setups that TimeRecipe outperforms SOTA (Paper Section 4.1.1).

For data properties calculation, please follow: [[Code](https://github.com/decisionintelligence/TFB/tree/master/characteristics_extractor)], [[Setup(en)](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Readme_en.md)], [[Setup(cn)](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Readme_chn.md)].

## Contact

If you have any questions about the code, please contact Zhiyuan Zhao at `leozhao1997[at]gatech[dot]edu`.


## Acknowledgement

If you find our work useful, please cite our work:

```
[TBD]
```

This work also builds on previous works, please consider cite these works properly.

Time Series Library (TSLib). [[Code](https://github.com/thuml/Time-Series-Library)]

TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods. [[Paper](https://arxiv.org/abs/2403.20150)][[Code](https://github.com/decisionintelligence/TFB)]

