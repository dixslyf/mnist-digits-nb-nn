import matplotlib.pyplot as plt

from assignment.data_loaders import NBDataLoader


def cli_entry(data_dir) -> int:
    train_data_loader = NBDataLoader(data_dir, mode="train")
    x_train, y_train = train_data_loader.load()

    val_data_loader = NBDataLoader(data_dir, mode="val")
    x_val, y_val = val_data_loader.load()

    test_data_loader = NBDataLoader(data_dir, mode="test")
    x_test, y_test = test_data_loader.load()

    # Display shapes of the data
    print("Shapes of the data:")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Visualise an image
    image_index = 4
    plt.imshow(x_train[image_index], cmap="gray")
    plt.title(f"Label: {y_train[image_index]}")
    plt.show()

    return 0
