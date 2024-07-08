import json

import pandas as pd

from descriptors import Descriptors


def get_unique_smiles(save=False):

    df = pd.read_csv("../data/datasets/modified/original_data/ready_to_go_2.csv")
    sense = [json.loads(i) for i in df.Sense]
    antisense = [json.loads(i) for i in df.AntiSense]
    seqs = []
    for seq in sense + antisense:
        for nuc in seq:
            if nuc not in seqs:
                seqs.append(nuc)

    if save is True:
        filename = "unique_smiles.csv"
        filepath = "../data/datasets/modified/original_data/"
        unique_smiles_df = pd.DataFrame(seqs, index=None)
        unique_smiles_df.to_csv(filepath + filename, index=False, header=False)

    return seqs


def load_unique_smiles():
    df = pd.read_csv("../data/datasets/modified/original_data/unique_smiles.csv")
    return df.UniqueSmiles.to_list()


def get_unique_descriptors(descriptors, save=False):

    unique_smiles = load_unique_smiles()
    calculated_descriptors = {}
    for i, smiles in enumerate(unique_smiles, 1):
        print(f"{i}. {smiles}")
        calculated_descriptors[smiles] = descriptors.descriptors_from_smiles(smiles)
        print(calculated_descriptors[smiles], end='\n\n')

    df = pd.DataFrame(columns=["smiles"] + descriptors.descriptor_names)
    for smiles, desc_list in calculated_descriptors.items():
        df.loc[df.shape[0]] = [smiles] + desc_list

    if save is True:
        filepath = "../data/datasets/modified/desciptors/unique_smiles/"
        filename = f"unique_{descriptors.__name__.lower()}_descriptors.csv"
        df.to_csv(filepath + filename, index=False)


if __name__ == "__main__":
    # get_unique_descriptors(Descriptors.PyBioMed, save=True)
    ...
