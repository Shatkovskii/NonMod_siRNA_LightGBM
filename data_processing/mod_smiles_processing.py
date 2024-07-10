import json

import pandas as pd
import requests
from bs4 import BeautifulSoup

from descriptors import Descriptors
from data_processing.force_field import ForceField
from data_processing.depscriptors_parsing import SCBDD


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
        filepath = "../data/datasets/modified/descriptors/unique_smiles/"
        filename = f"unique_{descriptors.__name__.lower()}_descriptors.csv"
        df.to_csv(filepath + filename, index=False)


def parse_unique_descriptors(descriptors, settings=None):

    unique_smiles = load_unique_smiles()

    descriptors_allowed = {SCBDD.Chemopy: 'Chemopy',
                           SCBDD.CDK: 'CDK',
                           SCBDD.RDKit: 'RDKit',
                           SCBDD.Pybel: 'Pybel',
                           SCBDD.BlueDesc: 'BlueDesc',
                           SCBDD.PaDEL: 'PaDEL'}

    url = 'http://www.scbdd.com/%s/index/' % descriptors
    filename = '../data/datasets/modified/descriptors/unique_smiles/unique_%s_descriptors.csv' % descriptors_allowed[descriptors].lower()
    settings = settings if settings is not None else {}

    force_field = settings.get("force_field", None)
    check_box_d = settings.get("check_box_d", "3D")

    data = {'Smiles': 'c1ccccc1'}
    if descriptors in SCBDD.Chemopy:
        data['check_box_d'] = check_box_d

    elif descriptors == SCBDD.BlueDesc:
        data['forcefield'] = force_field
        filename = filename.replace('.csv', '_%s.csv' % force_field)

    elif descriptors == SCBDD.PaDEL:

        data['check_box_d'] = check_box_d
        if force_field in [ForceField.mm2, ForceField.mmff94]:
            data['convert3d'] = 'Yes (use {} forcefield)'.format(force_field.upper())
            filename = filename.replace('.csv', '_%s.csv' % force_field)
        else:
            data['convert3d'] = 'No'
            filename = filename.replace('.csv', '_none_ff.csv')

    # Parse descriptors names
    response = requests.post(url, data=data)
    soup = BeautifulSoup(response.text, 'html.parser')
    info_table = soup.find('table', class_='table table-bordered table-condensed')
    desc_lines = info_table.find_all('tr', class_='altrow')
    desc_names = [line.find_all('td')[1].get_text(strip=True) for line in desc_lines]

    # Create blank dataframe
    desc_df = pd.DataFrame(columns=desc_names, index=unique_smiles)

    # Parse descriptors from monomer_dict
    for i, smiles in enumerate(unique_smiles, 1):
        data['Smiles'] = smiles
        response = requests.post(url, data=data)
        soup = BeautifulSoup(response.text, 'html.parser')

        try:
            info_table = soup.find('table', class_='table table-bordered table-condensed')
            desc_lines = info_table.find_all('tr', class_='altrow')
            for line in desc_lines:
                columns = [col.get_text(strip=True) for col in line.find_all('td')]
                desc_name = columns[1]
                desc_value = columns[2]
                desc_df.loc[smiles, desc_name] = desc_value
            status = '\033[92m' + 'done' + '\033[0m'
        except:
            status = '\033[93m' + 'descriptors not found' + '\033[0m'
            desc_df = desc_df[desc_df.index != smiles]

        # Save parsed descriptors dataset
        desc_df.to_csv(filename, index=True)

        log = '{}/{}. {} '.format(i, len(unique_smiles), smiles)
        print(log + '\033[92m' + status + '\033[0m')
    print()


if __name__ == "__main__":
    # get_unique_descriptors(Descriptors.CDK, save=True)
    parse_unique_descriptors(SCBDD.Chemopy)
    ...
