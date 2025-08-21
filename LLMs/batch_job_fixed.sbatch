#!/bin/bash
#SBATCH --job-name=openrouter_prompts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/openrouter_prompts_%j.out
#SBATCH --error=logs/openrouter_prompts_%j.err

# =============================================================================
# OPENROUTER PROMPT COMPARISON EXPERIMENT
# =============================================================================

cleanup() {
    echo "=== CLEANUP STARTED at $(date) ==="

    if [ ! -z "$MAIN_PID" ]; then
        echo "Killing main process $MAIN_PID"
        kill -TERM $MAIN_PID 2>/dev/null || true
        sleep 5
        kill -KILL $MAIN_PID 2>/dev/null || true
    fi

    find $PROJECT_DIR -name "*.pyc" -delete 2>/dev/null || true
    find $PROJECT_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    if [ ! -z "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi

    echo "=== CLEANUP COMPLETED at $(date) ==="
}

trap cleanup EXIT SIGTERM SIGINT SIGQUIT

# =============================================================================
# JOB SETUP
# =============================================================================

echo "=== PROMPT COMPARISON EXPERIMENT STARTED at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Project setup - FIXED PATHS
BASE_DIR="$HOME/repos/HTR_PostProcesing_LLM/LLMs"
PROJECT_DIR="$BASE_DIR/src"
VENV_PATH="$BASE_DIR/.venv"
ENV_FILE="$BASE_DIR/.env"

mkdir -p logs

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Load environment variables BEFORE changing directory
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from: $ENV_FILE"
    export $(cat $ENV_FILE | grep -v '^#' | xargs)
else
    echo "❌ Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Now change to project directory
cd $PROJECT_DIR || exit 1

# Activate virtual environment
source $VENV_PATH/bin/activate || exit 1

# Verify OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY not set even after loading .env"
    echo "Please check your .env file contains: OPENROUTER_API_KEY=your_key_here"
    exit 1
else
    echo "✅ OPENROUTER_API_KEY is set"
fi

echo "=== ENVIRONMENT INFO ==="
echo "Base directory: $BASE_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# All OpenRouter models to test
MODELS=(
#    "qwen-2.5-vl-72b"
#    "phi-4"
#    "internalvl3"
#    "mistral"
    "gemini-2.5-pro"
#    "gpt-4.1-mini"
#    "claude-sonnet-4"
)

# All datasets
DATASETS=(
    "iam"
    "bentham"
    "washington"
)

# HTR Models
HTR_MODELS=(
    "Flor_model"
)

# All 4 prompts
PROMPT_METHODS=(
    "promptOR_1"
    "promptOR_2"
    "promptOR_3"
    "promptOR_4"
)

# All training sizes
TRAIN_SIZES=(
    "train_25"
    "train_50"
    "train_75"
    "train_100"
)

# Options
MAX_SAMPLES=""  # Set to "--max-samples 100" for testing
VERBOSE="--verbose"
DRY_RUN=""  # Set to "--dry-run" for testing

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=== STARTING PROMPT COMPARISON EXPERIMENTS ==="
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "HTR Models: ${HTR_MODELS[@]}"
echo "Prompts: ${PROMPT_METHODS[@]}"
echo "Train Sizes: ${TRAIN_SIZES[@]}"
echo ""

# Convert arrays to strings
MODELS_STR="${MODELS[@]}"
DATASETS_STR="${DATASETS[@]}"
HTR_MODELS_STR="${HTR_MODELS[@]}"
PROMPT_METHODS_STR="${PROMPT_METHODS[@]}"
TRAIN_SIZES_STR="${TRAIN_SIZES[@]}"

# Build command
COMMAND="python main_openrouter.py \
    --models $MODELS_STR \
    --datasets $DATASETS_STR \
    --htr-models $HTR_MODELS_STR \
    --prompt-methods $PROMPT_METHODS_STR \
    --train-sizes $TRAIN_SIZES_STR \
    $MAX_SAMPLES \
    $VERBOSE \
    $DRY_RUN"

echo "Executing command:"
echo "$COMMAND"
echo ""

$COMMAND &
MAIN_PID=$!

wait $MAIN_PID
EXIT_CODE=$?

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "=== EXPERIMENT SUMMARY ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiments completed successfully"

    echo ""
    echo "=== RESULTS LOCATION ==="
    echo "Results are saved in: ../results/{dataset}/{htr_model}/{model}/promptOR_{1-4}/{train_size}/"

    # Count result files
    RESULT_COUNT=$(find ../results -name "results_no_suggestions_*.json" -mtime -1 | wc -l)
    echo "New result files created: $RESULT_COUNT"
else
    echo "❌ Experiments failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
