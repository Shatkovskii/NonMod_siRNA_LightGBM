import pandas as pd


def load_unique_descriptors(descriptors):
    filepath = "../data/datasets/modified/desciptors/unique_smiles/"
    filename = f"unique_{descriptors.__name__.lower()}_descriptors.csv"
    df = pd.read_csv(filepath + filename)
    unique_desc = {}
    for i, line in df.iterrows():
        unique_desc[line.smiles] = line.to_list()[1:]
    return unique_desc
