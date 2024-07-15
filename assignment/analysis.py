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
    image_index = 4
    plt.imshow(X[image_index], cmap="gray")
    plt.title(f"Label: {y[image_index]}")
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

    return 0
