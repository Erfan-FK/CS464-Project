@echo off
echo Running CNN Results Visualization...
python visualize_results.py --results_csv results.csv --test_out test.out --class_names ..\features\class_names.txt --output_dir visualize
echo Done!
pause
