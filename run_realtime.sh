#!/bin/bash
python run_realtime.py \
  --checkpoint model/stage1/model-15000 \
  --mode test \
  --result_dir results/stage1/ \
  --begin 0 \
  --end 50
