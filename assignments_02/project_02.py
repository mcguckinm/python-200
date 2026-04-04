import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


os.makedirs("outputs", exist_ok=True)

# Task 1:

df = pd.read_csv("../assignments/resources/student_performance_math.csv", sep=";")

print("\nTask 1")

print("Shape: ", df.shape)
print("\nFirst five rows: ")
print(df.head())
print("\nDataTypes: ")
print(df.dtypes)


plt.figure()
plt.hist(df["G3"], bins=np.arange(-0.5, 21.5, 1), edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Count")
plt.savefig("outputs/g3_distribution.png")
plt.close()

#Task 2

print("\nTask 2")

print("Original shape: ", df.shape)
df_clean = df[df["G3"] != 0].copy()

print("Filtered shape: ",df_clean.shape)
print("Rows removed: ", len(df) - len(df_clean))

#Removing G3=0 rows as those represent students who did not take the exam, not the ones who actually earned a 0.
#Keeping them gives a false assumption of data about academic performance that would be incorrect

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Correlation absences vs G3 (original): ", corr_original)
print("Correlation absences vs G3 (filtered): ", corr_filtered)

# Filtering data changes the amount of issues we have in G3=0 rows. Students who were absent for the final exam having them mixed will distort
#the final number with making sure you have students who did or did not take the exam. 

#task 3

