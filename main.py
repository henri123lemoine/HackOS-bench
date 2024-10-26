import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from src.config import PretrainedConfig
from src.models.ViT import PretrainedImageClassifier
from src.settings import MODELS_PATH


if __name__ == "__main__":
    checkpoint_path = MODELS_PATH / "vit" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model_config = PretrainedConfig(
        model_name="google/vit-base-patch16-224",
        model_class=ViTForImageClassification,
        processor_class=ViTImageProcessor,
        num_labels=2,
    )
    model = PretrainedImageClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    from betamark.bicycle import run_eval, run_validation
    EVAL = False
    if EVAL:
        # from src.eval import run_eval
        results = run_eval(user_func=lambda x: 1 - model.predict(x))
    else:
        results = run_validation(user_func=lambda x: 1 - model.predict(x))
    print(f"Evaluation Results: {results}")
