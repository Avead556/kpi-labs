import viewer
import scenes

if __name__ == "__main__":
    count = sum(e["config"]["features"] == 2 for e in scenes._Wine().experiments)
    viewer.main("boundary", "WineNet: межі рішень", "experiment", count, 5000, 50, __doc__)
