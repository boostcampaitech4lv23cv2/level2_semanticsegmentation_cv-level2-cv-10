python /opt/ml/input/level2_semanticsegmentation_cv-level2-cv-10/utils/ensemble.py \
--file_list \
"['/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-10/csv_final_v2/7062_pseudo.csv',
'/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-10/csv_final_v2/7156_PAN_swinl.csv',
'/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-10/csv_final_v2/knet_soup_v2.csv',
'/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-10/csv_final_v2/7348_ensemble.csv'
]" \
--output_file_name 'test6' \
--method 'weighted_majority' \
--Weight '[1, 1, 1.5, 1]'