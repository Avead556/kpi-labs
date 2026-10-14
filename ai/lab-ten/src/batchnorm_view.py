import viewer
import scenes

if __name__ == "__main__":
    viewer.main(scenes.BatchNormScene, "BatchNorm: з і без",
                {"epoch": (20, 1, 20), "pair": (0, 0, 2)}, __doc__)
