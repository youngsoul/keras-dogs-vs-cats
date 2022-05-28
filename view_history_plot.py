import pandas as pd


def main():
    losses = pd.read_csv("./history/fast_feature_extraction_cell_images.csv")
    print(losses.head())
    losses[['accuracy', 'val_accuracy']].plot()
    losses[['loss', 'val_loss']].plot()
if __name__ == '__main__':
    main()