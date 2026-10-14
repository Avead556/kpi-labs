import viewer
import scenes

if __name__ == "__main__":
    viewer.run(scenes.CurvesScene, "ПР №11: криві навчання",
               {"dataset": 0, "epoch": 10}, "epoch", (1, 10), __doc__)
