from src.models.ViT import create_vit_classifier
from src.dataset.bicycle import run_eval
from src.settings import MODELS_PATH


if __name__ == "__main__":
    # Create and load model
    model = create_vit_classifier()
    try:
        model.load(MODELS_PATH / "best_vit_model.pth")
    except:
        print("No saved model found, using untrained model")

    # Create prediction function wrapper
    def predict_bicycle(x):
        return model.predict(x)

    # Run evaluation
    score = run_eval(user_func=predict_bicycle)
    print(f"{score = }")
