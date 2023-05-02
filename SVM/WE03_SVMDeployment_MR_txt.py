import pickle
import pandas as pd

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the progrintam and print the current working directory.
# import os
# exit(os.getcwd())

riding = pickle.load(open("svm_poly.pkl","rb"))

print("\n*****************************************************")
print("* The prediction model for riding mover dataset as follows *")
print("*****************************************************\n")
Income= float(input("Enter the income "))
Lot_Size=float(input("Enter the lot size of individual"))
data = pd.DataFrame({'Income': [Income],'Lot_Size':[Lot_Size]})
output = riding.predict(data)
probability = riding.predict_proba(data)
predictions = ('Nonowner', 'Owner')
print(f"\nThe riding model indicates the probability of predictions at the {probability[0][1]:.4f}, which shows that the individual is {predictions[output[0]]}.\n")