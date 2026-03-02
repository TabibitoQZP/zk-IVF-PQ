# python -m tests.msmacro_broad_test \
#   --config data/msmacro/broad_test/config.json \
#   --out-dir data/msmacro/broad_test/out_full_l2 \
#   --plot-format pdf \
#   --plot

python -m tests.msmacro_broad_test \
  --config data/msmacro/broad_test_qps_align/config.json \
  --out-dir data/msmacro/broad_test_qps_align/out_full_l2 \
  --save-report data/msmacro/broad_test_qps_align/report.json \
  --plot-dir data/msmacro/broad_test_qps_align/plots \
  --plot-format pdf \
  --plot
