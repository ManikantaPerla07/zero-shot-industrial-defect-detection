import torch


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()


def print_device_info() -> None:
    """Print information about the selected computation device."""
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available - using CPU")


if __name__ == "__main__":
    print_device_info()
