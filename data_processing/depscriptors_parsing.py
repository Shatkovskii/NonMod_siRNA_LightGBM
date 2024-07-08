import pandas as pd
import requests
from bs4 import BeautifulSoup

from data_processing.sequant_funcs import aa_dict
from data_processing.force_field import ForceField


class SCBDD:

    """All the descriptors options one can find on: http://www.scbdd.com/chemdes/"""

    Chemopy = 'chemopy_desc'
    CDK = 'cdk_desc'
    RDKit = 'rdk_desc'
    Pybel = 'pybel_desc'
    BlueDesc = 'blue_desc'
    PaDEL = 'padel_desc'


def parse_scbdd(descriptors, monomer_dict=None, force_field=None):
    if monomer_dict is None:
        monomer_dict = aa_dict

    descriptors_allowed = {SCBDD.Chemopy: 'Chemopy',
                           SCBDD.CDK: 'CDK',
                           SCBDD.RDKit: 'RDKit',
                           SCBDD.Pybel: 'Pybel',
                           SCBDD.BlueDesc: 'BlueDesc',
                           SCBDD.PaDEL: 'PaDEL'}

    if descriptors not in descriptors_allowed:
        log = ('Parse SCBDD Error:  unknown descriptors type\n'
               '\t\texpected:   SCBDD.Chemopy, SCBDD.CDK, SCBDD.RDKit, SCBDD.Pybel, SCBDD.BlueDesc or SCBDD.PaDEL\n'
               f'\t\tgot:        {descriptors}')
        print('\033[91m' + log + '\033[0m')
        exit(1)

    elif descriptors == SCBDD.BlueDesc and force_field not in [
        ForceField.mmff94, ForceField.mmff94s, ForceField.ghemical, ForceField.gaff, ForceField.uff
    ]:
        log = ('Parse SCBDD Error:  unknown force field\n'
               '\t\texpected:   ForceField.mmff94, ForceField.mmff94s, ForceField.ghemical, ForceField.gaff or '
               'ForceField.uff\n'
               f'\t\tgot:        {force_field}')
        print('\033[91m' + log + '\033[0m')
        exit(1)

    elif descriptors == SCBDD.PaDEL and force_field not in [ForceField.mmff94, ForceField.mm2, None]:
        log = ('Parse SCBDD Error:  unknown force field\n'
               '\t\texpected:   ForceField.mmff94, ForceField.mm2 or None\n'
               f'\t\tgot:        {force_field}')
        print('\033[91m' + log + '\033[0m')
        exit(1)

    else:
        print('Start parsing {} descriptors'.format(descriptors_allowed[descriptors]))
        if descriptors in [SCBDD.BlueDesc, SCBDD.PaDEL]:
            print('Force Field:', force_field, end='\n\n')
        else:
            print()

    url = 'http://www.scbdd.com/%s/index/' % descriptors
    filename = 'Datasets/Descriptors/%s_descriptors.xlsx' % descriptors_allowed[descriptors].lower()

    data = {'Smiles': 'c1ccccc1'}
    if descriptors in SCBDD.Chemopy:
        data['check_box_d'] = '3D'

    elif descriptors == SCBDD.BlueDesc:
        data['forcefield'] = force_field
        filename = filename.replace('.xlsx', '_%s.xlsx' % force_field)

    elif descriptors == SCBDD.PaDEL:

        data['check_box_d'] = '3D'
        if force_field in [ForceField.mm2, ForceField.mmff94]:
            data['convert3d'] = 'Yes (use {} forcefield)'.format(force_field.upper())
            filename = filename.replace('.xlsx', '_%s.xlsx' % force_field)
        else:
            data['convert3d'] = 'No'
            filename = filename.replace('.xlsx', '_none_ff.xlsx')

    # Parse descriptors names
    response = requests.post(url, data=data)
    soup = BeautifulSoup(response.text, 'html.parser')
    info_table = soup.find('table', class_='table table-bordered table-condensed')
    desc_lines = info_table.find_all('tr', class_='altrow')
    desc_names = [line.find_all('td')[1].get_text(strip=True) for line in desc_lines]

    # Create blank dataframe
    desc_df = pd.DataFrame(columns=desc_names, index=monomer_dict.keys())

    # Parse descriptors from monomer_dict
    for monomer_name, smiles in monomer_dict.items():
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
                desc_df.loc[monomer_name, desc_name] = desc_value
            status = '\033[92m' + 'done' + '\033[0m'
        except:
            status = '\033[93m' + 'descriptors not found' + '\033[0m'
            desc_df = desc_df[desc_df.index != monomer_name]

        # Save parsed descriptors dataset
        writer = pd.ExcelWriter(filename)
        desc_df.to_excel(writer, index=True)
        writer._save()

        log = '{}/{}. {} '.format(list(monomer_dict.keys()).index(monomer_name)+1, len(monomer_dict), monomer_name)
        print(log + '\033[92m' + status + '\033[0m')
    print()
