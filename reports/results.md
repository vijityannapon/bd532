# Training results

Data: **home_credit**, 307,511 rows, default rate 8.07%

## Model versions (validation set)

| name                   |   roc_auc |   pr_auc |     ks |   brier |   recall |   precision |   flag_rate |   expected_cost |   gender_fpr_diff |   gender_disparate_impact |   latency_ms_single |   model_size_mb |   fit_seconds |
|:-----------------------|----------:|---------:|-------:|--------:|---------:|------------:|------------:|----------------:|------------------:|--------------------------:|--------------------:|----------------:|--------------:|
| v1_logreg_basic        |    0.7426 |   0.2273 | 0.3542 |  0.0687 |   0.3518 |      0.2469 |       0.115 |          0.3483 |            0.0486 |                    0.9319 |              1.7357 |          0.0088 |        1.7241 |
| v2_logreg_engineered   |    0.7437 |   0.23   | 0.3569 |  0.0686 |   0.3569 |      0.2505 |       0.115 |          0.3458 |            0.0443 |                    0.9376 |              1.8925 |          0.0099 |        2.0796 |
| v3_random_forest       |    0.749  |   0.2348 | 0.3691 |  0.1832 |   0.3298 |      0.2662 |       0.1   |          0.3439 |            0.0372 |                    0.9463 |             26.3525 |         37.3362 |       13.0002 |
| v4_hgb_default         |    0.7579 |   0.2466 | 0.3815 |  0.0677 |   0.3561 |      0.2737 |       0.105 |          0.3362 |            0.0428 |                    0.9385 |              8.6214 |          0.3837 |        5.5473 |
| v5_hgb_tuned           |    0.7607 |   0.2484 | 0.3827 |  0.0676 |   0.3818 |      0.268  |       0.115 |          0.3337 |            0.043  |                    0.9379 |             13.8421 |          0.6196 |        7.644  |
| v6_hgb_tuned_reweighed |    0.7612 |   0.2458 | 0.3888 |  0.0677 |   0.3888 |      0.2616 |       0.12  |          0.3353 |            0.016  |                    0.9687 |             31.506  |          1.4656 |       21.2772 |

## Selection

Calibration gate (Brier <= 0.0744) excluded ['v3_random_forest']. Highest eligible validation ROC-AUC is 0.7612 (v6_hgb_tuned_reweighed). Models within 0.003 AUC are treated as equivalent: ['v5_hgb_tuned', 'v6_hgb_tuned_reweighed']. Among them the one with the smallest gender FPR gap (then lowest latency) is selected: **v6_hgb_tuned_reweighed**.

5-fold CV ROC-AUC on train: 0.7561 ± 0.0052

## Test set (held out)

|               |      value |
|:--------------|-----------:|
| roc_auc       |     0.7642 |
| pr_auc        |     0.2457 |
| ks            |     0.3987 |
| brier         |     0.0677 |
| precision     |     0.2529 |
| recall        |     0.3797 |
| f1            |     0.3036 |
| flag_rate     |     0.1212 |
| expected_cost |     0.3409 |
| tp            |  1414      |
| fp            |  4177      |
| tn            | 38226      |
| fn            |  2310      |

## Decision bands (test)

| band      |   share |   actual_default_rate |
|:----------|--------:|----------------------:|
| HIGH_RISK |  0.1212 |                0.2529 |
| LOW_RISK  |  0.6773 |                0.0373 |
| REVIEW    |  0.2014 |                0.1233 |

## Fairness - gender (test)

| group   |     n |   actual_default_rate |   mean_predicted_pd |   approval_rate |   tpr_defaulters_caught |   fpr_good_flagged |    auc |
|:--------|------:|----------------------:|--------------------:|----------------:|------------------------:|-------------------:|-------:|
| F       | 30429 |                0.0696 |              0.0766 |          0.8923 |                  0.3536 |             0.0893 | 0.7608 |
| M       | 15698 |                0.1023 |              0.0876 |          0.8526 |                  0.4141 |             0.117  | 0.7632 |

|                         |   value |
|:------------------------|--------:|
| demographic_parity_diff |  0.0397 |
| disparate_impact_ratio  |  0.9555 |
| equal_opportunity_diff  |  0.0604 |
| fpr_diff                |  0.0277 |
| auc_diff                |  0.0025 |

## Fairness - age_group (test)

| group   |     n |   actual_default_rate |   mean_predicted_pd |   approval_rate |   tpr_defaulters_caught |   fpr_good_flagged |    auc |
|:--------|------:|----------------------:|--------------------:|----------------:|------------------------:|-------------------:|-------:|
| 30-50   | 23838 |                0.0868 |              0.0858 |          0.8588 |                  0.4132 |             0.1153 | 0.7702 |
| <30     |  6720 |                0.1095 |              0.1116 |          0.7866 |                  0.4864 |             0.1798 | 0.7392 |
| >50     | 15569 |                0.059  |              0.0585 |          0.9492 |                  0.2187 |             0.0403 | 0.7417 |

|                         |   value |
|:------------------------|--------:|
| demographic_parity_diff |  0.1626 |
| disparate_impact_ratio  |  0.8287 |
| equal_opportunity_diff  |  0.2677 |
| fpr_diff                |  0.1395 |
| auc_diff                |  0.031  |

## Global importance

|                      |   mean_abs_shap |   permutation_auc_drop |
|:---------------------|----------------:|-----------------------:|
| ext_source_mean      |          0.5146 |                 0.0981 |
| payment_rate         |          0.1831 |                 0.0281 |
| ext_source_3         |          0.1117 |                 0.01   |
| own_car              |          0.1021 |                 0.004  |
| goods_credit_ratio   |          0.0916 |                 0.0054 |
| education            |          0.0874 |                 0.0065 |
| goods_price          |          0.0785 |                 0.0062 |
| age_years            |          0.0724 |                 0.0086 |
| annuity              |          0.0663 |                 0.0017 |
| years_employed       |          0.061  |                 0.0025 |
| family_status        |          0.0595 |                 0.0011 |
| ext_source_2         |          0.0561 |                 0.0039 |
| contract_type        |          0.0522 |                 0.0021 |
| annuity_income_ratio |          0.0514 |                 0.0034 |
| ext_source_missing   |          0.0483 |                 0.0007 |
| employed_age_ratio   |          0.0453 |                 0.0019 |
| ext_source_1         |          0.0421 |                 0.0045 |
| region_rating        |          0.0359 |                -0.0009 |
| occupation           |          0.0265 |                 0.0001 |
| income_type          |          0.0229 |                 0.0004 |
| credit_income_ratio  |          0.0213 |                 0.0004 |
| credit_amount        |          0.0212 |                 0.0017 |
| income               |          0.0201 |                 0.0013 |
| own_realty           |          0.0148 |                -0.0002 |
| housing_type         |          0.008  |                -0.0007 |
| children             |          0.0057 |                 0.0005 |
| family_members       |          0.0047 |                 0.0005 |
| employment_missing   |          0.0008 |                -0.0002 |

## Example explanations (3 test cases)

### low_risk_case: PD=1.2%, score=627, band=LOW_RISK, actual=0

* Average external credit score = 0.701 decreases default risk (SHAP -0.778)
* Payment rate (installment / loan) = 0.098 decreases default risk (SHAP -0.714)
* Family status = Single / not married increases default risk (SHAP +0.126)
* Age = 61.0 yrs decreases default risk (SHAP -0.119)
* Region risk rating (1-3) = 1 decreases default risk (SHAP -0.102)

### borderline_case: PD=16.4%, score=468, band=HIGH_RISK, actual=0

* Average external credit score = 0.337 increases default risk (SHAP +0.798)
* Payment rate (installment / loan) = 0.0512 increases default risk (SHAP +0.231)
* External credit score 1 = 0.141 increases default risk (SHAP +0.184)
* Education = Higher education decreases default risk (SHAP -0.174)
* Region risk rating (1-3) = 3 increases default risk (SHAP +0.139)

### high_risk_case: PD=33.6%, score=414, band=HIGH_RISK, actual=0

* Average external credit score = 0.26 increases default risk (SHAP +0.798)
* Goods price / loan ratio = 0.703 increases default risk (SHAP +0.339)
* Payment rate (installment / loan) = 0.0468 increases default risk (SHAP +0.236)
* Age = 24.3 yrs decreases default risk (SHAP -0.142)
* Price of goods / collateral = 450,000 increases default risk (SHAP +0.136)
