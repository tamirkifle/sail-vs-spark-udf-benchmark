# SLURM Runs

These scripts are for cluster runs where login nodes may have internet but
compute nodes often do not.

## CPU Real Mode

`cpu_real` uses in-process Hugging Face Transformers generation on CPU. It does
not install or start vLLM.

From the repo root on the cluster:

```bash
module load anaconda3/2024.06 || true
module load OpenJDK/22.0.2 || true

scripts/setup_env.sh --mode cpu_real --venv .venv
mkdir -p logs
sbatch slurm/submit_cpu_real_models.sh
```

Spark requires Java 17 or newer. `submit_cpu_real_models.sh` prints the selected
`java` path and `java -version`, then exits before running benchmarks if Java is
too old.

The submit script also prints the `transformers` version and import path. If it
reports that `AutoModelForCausalLM` is missing, refresh the venv on a node with
package access:

```bash
.venv/bin/python -m pip install --upgrade \
  "transformers>=4.51.0" "sentence-transformers>=3.0.0" "accelerate>=0.26.0"
```

## Offline Compute Nodes

By default, `submit_cpu_real_models.sh` sets `FORCE_SYNTHETIC=1` so dataset prep
does not contact Hugging Face from a compute node.

To run with real UltraFeedback prompts on an offline compute node, pre-stage the
prepared parquet on a login node that has internet:

```bash
module load anaconda3/2024.06 || true

# Use the same venv/cache locations as the submit job.
scripts/setup_env.sh --mode cpu_real --venv .venv
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$PWD/models"
export MODELS_DIR="$PWD/models"

# Download all non-mock cpu_real models, including the Transformers generator.
.venv/bin/python scripts/download_models.py \
  --config config/cpu_real.yaml \
  --models-dir "$MODELS_DIR"

# Download/prepare the dataset once and write data/cpu_real/prompts.parquet.
.venv/bin/python scripts/prep_dataset.py --config config/cpu_real.yaml
```

Then submit the compute job while reusing the staged parquet:

```bash
SKIP_DATASET_PREP=1 sbatch slurm/submit_cpu_real_models.sh
```

`SKIP_DATASET_PREP=1` tells `scripts/run_benchmark.sh` not to delete and rebuild
`data/cpu_real/prompts.parquet`. The job still uses the staged model/cache
directories configured by the submit script:

```text
HF_HOME=$REPO_DIR/.cache/huggingface
HF_HUB_CACHE=$REPO_DIR/.cache/huggingface/hub
SENTENCE_TRANSFORMERS_HOME=$REPO_DIR/models
MODELS_DIR=$REPO_DIR/models
```

If compute nodes also have internet and you want the job to perform dataset prep
itself, submit with:

```bash
FORCE_SYNTHETIC=0 sbatch slurm/submit_cpu_real_models.sh
```

That path is not recommended for offline partitions because Hugging Face
streaming can block for a long time before failing.
