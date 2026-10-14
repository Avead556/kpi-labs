import viewer
import scenes

if __name__ == "__main__":
    count = len(scenes._Wine().experiments)
    viewer.main("sweep", "WineNet: порівняння модифікацій", "experiment", count, 5000, 50,
                __doc__)
