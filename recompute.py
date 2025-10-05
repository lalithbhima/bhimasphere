import pandas as pd

# Skip commented metadata lines ("#") at top
koi = pd.read_csv("kepler_objects_of_interest.csv", comment="#")
print(koi.columns[:100])  # sanity check
