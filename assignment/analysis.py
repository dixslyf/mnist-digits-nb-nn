import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def display_pixel_dists(X):
    rows = 28
    cols = 28
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        plt.hist(X[:, row, col], bins=20, histtype="step", stacked=True)
    plt.title("Pixel distributions")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()


def display_pca_dists(X, random_state):
    rows = 7
    cols = 7

    n_components = rows * cols
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X.reshape((X.shape[0], -1)))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
    for idx in range(n_components):
        ax = axes[idx // cols, idx % cols]
        ax.hist(X_pca[:, idx], bins=20, density=True)
        ax.set_xlabel("1st principal component value")
        ax.set_ylabel("Frequency density")
    plt.tight_layout()
    plt.show()


def cli_entry(data_dir, analysis, random_state) -> int:
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
            display_pixel_dists(X_train)
        case "pca-distributions":
            display_pca_dists(X_train, random_state)

    return 0
