from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from PyBioMed.PyMolecule import connectivity
from CDK_pywrapper import CDK

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sequant_funcs import aa_dict, seq_to_matrix_, SeQuant_encoding, generate_latent_representations
from depscriptors_parsing import SCBDD, parse_scbdd


class Rdkit:

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)

        descriptors_set = np.empty((0, num_descriptors), float)

        for _, value in monomer_dict_.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = np.array(get_descriptors.ComputeProperties(molecule))
            descriptors = descriptors.reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)
        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=96):

        descriptors_set = Rdkit.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        encoded_senses = SeQuant_encoding(sequences_list=sequences_list,
                                          polymer_type='RNA',
                                          descriptors=descriptors_set,
                                          num=96)

        x_senses = generate_latent_representations(sequences_list=encoded_senses,
                                                   sequant_encoded_sequences=encoded_senses,
                                                   polymer_type='RNA',
                                                   path_to_model_folder='Models/nucleic_acids')

        return x_senses


class PyBioMed:

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        # Get names of PyBioMed descriptors, obtain their amount
        descriptor_names = list(connectivity._connectivity.keys())
        num_descriptors = len(descriptor_names)

        # Create empty array where monomer descriptors will be kept
        descriptors_set = np.empty((0, num_descriptors), float)

        for _, value in monomer_dict_.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = list(connectivity.GetConnectivity(molecule).values())
            descriptors = np.array(descriptors)
            descriptors = descriptors.reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)
        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = PyBioMed.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class CDK_:

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        smiles_list = list(monomer_dict_.values())
        mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]

        cdk = CDK()
        desc_df = cdk.calculate(mols)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)
        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = CDK_.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class Mordred:
    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):
        """
        Mordred не завёлся локально, но нормально заработал в google.colab, поэтому просто сгенерил дескрипторы там.
        Дескрипторы уже нормализованы на (-1, 1). Код:

                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import MinMaxScaler

                smiles_list = list(aa_dict.values())
                mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

                calc = Calculator(descriptors, ignore_3D=False)
                desc_df = calc.pandas(mols)

                descriptor_names = desc_df.columns.to_list()
                descriptors_set = [line.values for _, line in desc_df.iterrows()]
                descriptors_set = np.array(descriptors_set)

                normalize = (-1, 1)
                sc = MinMaxScaler(feature_range=normalize)
                scaled_array = sc.fit_transform(descriptors_set)
                df = pd.DataFrame(scaled_array, columns=descriptor_names, index=list(aa_dict.keys()))
        """

        return pd.read_excel('Datasets/Descriptors/mordred_descriptors.xlsx', index_col=0)

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = Mordred.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class Chemopy:
    @staticmethod
    def parse_descriptors():
        """ Библиотека chemopy2 требует для работы библиотеку openbabel, её трудно установить с ходу,
        поэтому решил пока что просто запарсить дескрипторы с сайта http://www.scbdd.com/chemopy_desc/index/
        см. модуль descriptors_parsing.py
        """

        parse_scbdd(descriptors=SCBDD.Chemopy)

    @staticmethod
    def generate_descriptors(normalize: tuple = (-1, 1), monomer_dict_: dict = None):

        """Chemopy дескрипторы для словаря мономеров aa_dict запаршены с сайта с сайта:
        http://www.scbdd.com/chemopy_desc/index/
        Здесь мы просто их нормируем на диапазоне normalize
        """

        if monomer_dict_ is None:
            monomer_dict_ = aa_dict

        desc_df = pd.read_excel('Datasets/Descriptors/chemopy_descriptors.xlsx', index_col=0)

        descriptor_names = desc_df.columns.to_list()
        descriptors_set = [line.values for _, line in desc_df.iterrows()]
        descriptors_set = np.array(descriptors_set)

        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)

        return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys()))

    @staticmethod
    def encode_sequences(sequences_list, max_length=27):

        descriptors_set = Chemopy.generate_descriptors()

        if not isinstance(sequences_list, list):
            sequences_list = [sequences_list]

        container = []
        for sequence in sequences_list:
            seq_matrix = seq_to_matrix_(sequence=sequence,
                                        polymer_type='RNA',
                                        descriptors=descriptors_set,
                                        num=max_length)

            seq_matrix = np.array(seq_matrix).flatten()
            container.append(seq_matrix)

        return np.array(container)


class Descriptors:

    rdkit = Rdkit
    pybiomed = PyBioMed
    cdk = CDK_
    mordred = Mordred
    chemopy = Chemopy
