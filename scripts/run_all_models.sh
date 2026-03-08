#!/bin/bash
#===============================================================================

#===============================================================================




#===============================================================================

set -e  


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


USE_OPTUNA=true
N_TRIALS=50
VERSION="v1"
MODELS=("LogisticRegression" "RandomForest" "XGBoost" "LightGBM" "CatBoost")


while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            USE_OPTUNA=false
            shift
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --model)
            shift
            MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


echo "========================================"
echo "SpineIDS Clinical ML - Batch Training"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo "Use Optuna: $USE_OPTUNA"
echo "N Trials: $N_TRIALS"
echo "Version: $VERSION"
echo "Models: ${MODELS[*]}"
echo "========================================"


cd "$PROJECT_ROOT"


if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi


echo ""
echo "========================================"
echo "Step 1: Feature Selection"
echo "========================================"

python scripts/run_feature_selection.py


echo ""
echo "========================================"
echo "Step 2: Training Models"
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Training: $model"
    echo "----------------------------------------"
    
    if [ "$USE_OPTUNA" = true ]; then
        python scripts/run_pipeline.py \
            --model "$model" \
            --version "$VERSION" \
            --n-trials "$N_TRIALS" \
            --skip-feature-selection
    else
        python scripts/run_pipeline.py \
            --model "$model" \
            --version "$VERSION" \
            --no-optuna \
            --skip-feature-selection
    fi
    
    echo "$model training complete!"
done


echo ""
echo "========================================"
echo "Step 3: External Validation"
echo "========================================"

python scripts/run_external_eval.py --model all --version "$VERSION"


echo ""
echo "========================================"
echo "All models trained successfully!"
echo "========================================"
echo "Results saved to: $PROJECT_ROOT/outputs/"
echo ""


if [ -f "outputs/model_comparison_summary.csv" ]; then
    echo "Model Comparison Summary:"
    cat outputs/model_comparison_summary.csv
fi
