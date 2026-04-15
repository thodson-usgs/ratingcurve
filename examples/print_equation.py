"""Print the power-law rating equation from a fitted model.

Demonstrates how to extract and display the parametric equation
for a multi-segment power-law rating curve.
"""
import numpy as np
from ratingcurve.ratings import PowerLawRating
from ratingcurve import data

# Load sample data and fit a 2-segment model
df = data.load('green channel')
rating = PowerLawRating(segments=2)
rating.fit(df['stage'], df['q'], q_sigma=df['q_sigma'])

# Get the denormalized equation parameters
params = rating.equation()

# Print the equation
segments = len(params['b'])

print("Rating curve equation")
print("=====================")
print()
print("  ln(q) = a + sum(b[i] * ln(max(h - hs[i], 0) + ho[i]))")
print()
print("where ho[0] = 0 and ho[i] = 1 for i > 0")
print()
print(f"  a  = {params['a']:.4f}")
for i in range(segments):
    print(f"  b[{i}] = {params['b'][i]:.4f},  hs[{i}] = {params['hs'][i]:.4f}")
print()

# Build the full equation string
terms = []
for i in range(segments):
    ho = 0 if i == 0 else 1
    if ho == 0:
        terms.append(f"{params['b'][i]:.4f} * ln(max(h - {params['hs'][i]:.4f}, 0))")
    else:
        terms.append(f"{params['b'][i]:.4f} * ln(max(h - {params['hs'][i]:.4f}, 0) + 1)")

eq_str = f"ln(q) = {params['a']:.4f} + " + " + ".join(terms)
print(eq_str)
