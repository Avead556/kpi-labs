import viewer
import scenes

if __name__ == "__main__":
    viewer.main(scenes.PredictionsScene, "Передбачення: без BN / з BN",
                {"epoch": (20, 1, 20), "pair": (0, 0, 2), "page": (0, 0, 1)}, __doc__)
