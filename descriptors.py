from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from PyBioMed.PyMolecule import connectivity
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sequant_funcs import aa_dict, seq_to_matrix_, SeQuant_encoding, generate_latent_representations


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


class Descriptors:

    rdkit = Rdkit
    pybiomed = PyBioMed
