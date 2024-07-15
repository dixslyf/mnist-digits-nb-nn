import matplotlib.pyplot as plt

from assignment.data_loaders import NBDataLoader


def display_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
    print("x_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


def display_samples(X, y):
    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.imshow(X[idx], cmap="gray")
        ax.set_title(f"Label: {y[idx]}")
    plt.tight_layout()
    plt.show()


def display_pixel_dists(X, y):
    rows = 28
    cols = 28
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        plt.hist(X[:, row, col], bins=20, histtype="step", stacked=True)
        plt.title("Pixel distributions")
    plt.show()


def cli_entry(data_dir, analysis) -> int:
    train_data_loader = NBDataLoader(data_dir, mode="train")
    X_train, y_train = train_data_loader.load()

    val_data_loader = NBDataLoader(data_dir, mode="val")
    X_val, y_val = val_data_loader.load()

    test_data_loader = NBDataLoader(data_dir, mode="test")
    X_test, y_test = test_data_loader.load()

    match analysis:
        case "shapes":
            display_shapes(X_train, y_train, X_val, y_val, X_test, y_test)
        case "samples":
            display_samples(X_train, y_train)
        case "pixel-distributions":
            display_pixel_dists(X_train, y_train)

    return 0
