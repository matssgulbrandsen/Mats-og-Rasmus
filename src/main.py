from train import load_data, time_based_split, tune_models, save_models
from evaluate import run_evaluation
from preprocessing import filter_invalid_waybills
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "all_waybill_info_meituan_0322.csv"
    filtered_path = input_path.parent / f"filtered_{input_path.name}"

    filter_invalid_waybills(input_path, filtered_path)
    df = load_data(f"filtered_{input_path.name}")
    train, test = time_based_split(df)
    trained = tune_models(train, test)
    save_models(trained)
    run_evaluation(trained)