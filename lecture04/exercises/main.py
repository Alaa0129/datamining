#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This file is not to be used to be run directly. It is more to quote commands
for the .tex file. Can anyone live without those three? Not in data!"""

# Importing the libraries.
from typing import List

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rnd
import seaborn as sns  # Seaborn make some plots way easier!

# Theme for seaborn.
sns.set_theme()

# Check the directories in you code! Copy&Paste is not enough!
data = pd.read_csv("../data/penguins.csv")  # Load penguins.csv files
data2 = pd.read_csv("../data/penguins2.csv")  # The messed csv file.

# Check the data types of the columns.
penguins = sns.load_dataset("penguins")
sns.displot(penguins, x="flipper_length_mm", hue="species")


def random_rename(original_list: List):
    """This function messes with the original DataFrame. The function takes a
    list of names and randomly choses one."""
    rand_indexes = rnd.uniform(0, len(original_list))
    attributes = original_list[math.floor(rand_indexes)]
    return attributes


adelies = ["adelie","Adelie","Adelie."]  # Messed list for Adelies.
gentoos = ["Gentoo","gentoo","gento"]  # Messed list for Gentoos.
chinstraps = ["Chinstrap","chinstrap","chintrap"]  # Messed list for Chinstraps.

# Create a new DataFrame to mess with the species names.
penguins2 = sns.load_dataset("penguins")

# Routine to mess with the species names.
for x in penguins2.index:
    if penguins2.species[x] == "Adelie":
        penguins2.at[x, "species"] = random_rename(adelies)
    elif penguins2.species[x] == "Gentoo":
        penguins2.at[x, "species"] = random_rename(gentoos)
    elif penguins2.species[x] == "Chinstrap":
        penguins2.at[x, "species"] = random_rename(chinstraps)

# Routine to add NaNs to the data frame.
for x in list(penguins2):
    for i in penguins2[x].index:
        if rnd.uniform(0, 1) < 0.1:
            penguins2[x].at[i] = np.nan

# Routine to make some floats into strings.
for x in list(penguins2):
    for i in penguins2[x].index:
        if rnd.uniform(0, 1) < 0.05:
            penguins2[x].at[i] = str(penguins2[x].at[i])

# Save a messed DataFrame into a CSV file.
penguins2.to_csv(penguins2.csv, index= False)
