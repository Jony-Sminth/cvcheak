from tests.test_process_image import PreprocessedConfig, PreprocessedPredictor
import torch

config = PreprocessedConfig(
    confidence_threshold=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=4
)

predictor = PreprocessedPredictor(
    model_path="path/to/model_final.pth",
    config=config
)

results = predictor.process_preprocessed_folder(
    preprocessed_dir="data/preprocessed_val/",
    output_path="output/preprocessed_results.json",
    batch_size=4
)
    