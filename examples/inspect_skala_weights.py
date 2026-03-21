from skala.functional import load_functional


def main():
    print("Loading Skala functional...")
    try:
        # Load the Skala functional
        # This will download it from Hugging Face if not locally cached/available
        func = load_functional("skala")

        print("\nModel Weights Shapes:")
        print("-" * 50)

        # Count total parameters
        total_params = 0

        # Iterate over named parameters
        for name, param in func.named_parameters():
            print(f"{name}: {tuple(param.shape)}")
            total_params += param.numel()

        print("-" * 50)
        print(f"Total parameters: {total_params}")

    except Exception as e:
        print(f"Error loading model or accessing weights: {e}")


if __name__ == "__main__":
    main()
