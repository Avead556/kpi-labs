import viewer
import scenes

if __name__ == "__main__":
    viewer.main(scenes.ExperimentsScene, "Експерименти ПР №10",
                {"epoch": (20, 1, 20), "part": (1, 0, 4)}, __doc__)
