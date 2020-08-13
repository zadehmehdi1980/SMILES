

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from sklearn.model_selection import train_test_split


def features_ext(smile_string, radius=2, nBits=256):

    mols = Chem.rdmolfiles.MolFromSmiles(smile_string)
    fps = rdMolDescriptors.GetMorganFingerprintAsBitVect(mols, radius=radius, bitInfo= dict(), nBits=nBits)
    
    return np.array(fps)


def dataset_split(data, nBits=256):

    return train_test_split(data[:, :nBits], data[:, -1], stratify=data[:, -1], test_size=0.2, random_state=42)
