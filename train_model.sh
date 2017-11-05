#!/bin/sh
fp="/scratch/tr1312/decomp_attn/"
fp_train="data/snli_preprocess/train.hdf5"
fp_dev="data/snli_preprocess/val.hdf5"
fp_test="data/snli_preprocess/test.hdf5"
fp_w2v="data/snli_preprocess/glove.hdf5"
fp_log="SNLI-decomposable-attention-master/log/"
fp_model="SNLI-decomposable-attention-master/model_output/"
fp_code="code/train_baseline_snli.py"

python "$fp$fp_code" \
--train_file "$fp$fp_train" \
--dev_file "$fp$fp_dev" \
--test_file "$fp$fp_test" \
--w2v_file "$fp$fp_w2v" \
--log_dir "$fp$fp_log" \
--model_path "$fp$fp_model" \
--gpu_id 0 \
--embedding_size 300