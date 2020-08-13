

from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from sklearn.model_selection import train_test_split


def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )

def X_reshape(X):
    
    return X.reshape(X.shape[0], X.shape[1], 1)


def dataset_split(data, size=2048):

    X_train, X_test, y_train, y_test = train_test_split(data[:, :size], data[:, -1], stratify=data[:, -1], test_size=0.2, random_state=42)

    X_train = X_reshape(X_train)

    X_test = X_reshape(X_test)

    return X_train, X_test, y_train, y_test