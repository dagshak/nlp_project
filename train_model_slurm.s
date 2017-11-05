#!/bin/sh
#SBATCH --job-name=in_too_deep_preprocessor
#SBATCH --output=slurm_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:50:00
#SBATCH --mem=24GB
#SBATCH --mail-type=tr1312@nyu.edu  # email me when the job ends

module load h5py/intel/2.7.0rc2
module load pytorch/0.2.0_1
module load numpy/intel/1.13.1
module load pandas/intel/py2.7/0.20.3

fp="/scratch/tr1312/decomp_attn/"
fp_train="data/snli_preprocess/train.hdf5"
fp_dev="data/snli_preprocess/val.hdf5"
fp_test="data/snli_preprocess/test.hdf5"
fp_w2v="data/snli_preprocess/glove.hdf5"
fp_log="log/"
fp_model="model_output/"
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