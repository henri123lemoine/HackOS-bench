from src.models.ViT import create_vit_classifier
from src.dataset.bicycle import run_eval
from src.settings import MODELS_PATH


if __name__ == "__main__":
    # Create and load model
    model = create_vit_classifier()
    try:
        model.load(MODELS_PATH / "vit" / "best_model.pth")
    except:
        print("No saved model found, using untrained model")

    score = run_eval(user_func=lambda x: model.predict(x))
    print(f"{score = }")
