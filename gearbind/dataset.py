import os
import csv
import glob
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from copy import deepcopy

from Bio.PDB import PDBParser

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import core, data, datasets, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from gearbind import residue_constants


def bio_load_pdb(pdb):
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb)
    residues = [residue for residue in protein.get_residues()]
    residue_type = [data.Protein.residue2id.get(residue.get_resname(), 0) for residue in residues]
    chain_id = [data.Protein.alphabet2id.get(residue.get_parent().id, 0) for residue in residues]
    insertion_code = [data.Protein.alphabet2id.get(residue.full_id[3][2], -1) for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}
    # residue_feature = functional.one_hot(torch.as_tensor(residue_type), len(data.Protein.residue2id)+1)

    atoms = [atom for atom in protein.get_atoms()]
    atoms = [atom for atom in atoms if atom.get_name() in data.Protein.atom_name2id]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name,
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor, chain_id=chain_id,
                residue_number=residue_number, node_position=node_position, insertion_code=insertion_code, # residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])


@R.register("datasets.SKEMPI")
class SKEMPI(data.ProteinDataset):

    fname = "SKEMPI.zip"
    md5 = "2c54e2ae7cda20cc5dfb2f5ab2adb8af"
    processed_file = "skempi.pkl.gz"
    splits = ["split_0", "split_1", "split_2", "split_3", "split_4"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = os.path.join(path, self.fname)
        path = os.path.join(utils.extract(zip_file), "SKEMPI")
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            csv_files = []
            for split in self.splits:
                split_path = utils.extract(os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
                csv_files.append(os.path.join(path, "%s.csv" % split))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.load_annotation(csv_files)
            self.save_pickle(pkl_file, verbose=verbose)

        pdb_splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [pdb_splits.count(split) for split in self.splits]

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                protein, sequence = bio_load_pdb(pdb_file)
            else:
                protein, sequence = None, None
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(sequence)

    def load_annotation(self, csv_files):
        data_dict = {
            os.path.basename(pdb_file): (protein, pdb_file, sequence) \
                for pdb_file, protein, sequence in zip(self.pdb_files, self.data, self.sequences)
        }
        self.data = []
        self.pdb_files = []
        self.sequences = []

        for fname in csv_files:
            csv_file = open(fname, "r")
            reader = csv.reader(csv_file, delimiter=',')
            header = next(reader)
            mutation_id, chain_a_id, chain_b_id, wt_protein_id, mt_protein_id = \
                map(header.index, ["mutation", "chain_a", "chain_b", "wt_protein", "mt_protein"])
            ddG_id = header.index("ddG") if "ddG" in header else None

            for line in reader:
                mutations, chain_a, chain_b, _wild_type, _mutant = \
                    map(lambda i: line[i], [mutation_id, chain_a_id, chain_b_id, wt_protein_id, mt_protein_id])
                ddG = line[ddG_id] if ddG_id is not None else 0.0
                mutations = mutations.split(",")

                if _wild_type not in data_dict: continue
                wild_type = data_dict[_wild_type][0]
                with wild_type.node():
                    entity_a = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    for a in chain_a:
                        entity_a |= wild_type.chain_id == wild_type.alphabet2id[a]
                    wild_type.entity_a = entity_a[wild_type.atom2residue]

                    entity_b = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    for b in chain_b:
                        entity_b |= wild_type.chain_id == wild_type.alphabet2id[b]
                    wild_type.entity_b = entity_b[wild_type.atom2residue]

                    is_mutation = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    for m in mutations:
                        if m[-2].isalpha():
                            is_mutation |= \
                                (wild_type.chain_id == wild_type.alphabet2id[m[1]]) & \
                                (wild_type.residue_number == int(m[2:-2])) & \
                                (wild_type.insertion_code == wild_type.alphabet2id[m[-2]])
                        else:
                            is_mutation |= \
                                (wild_type.chain_id == wild_type.alphabet2id[m[1]]) & \
                                (wild_type.residue_number == int(m[2:-1]))
                    wild_type.is_mutation = is_mutation[wild_type.atom2residue]
                wild_type = wild_type.subgraph(wild_type.entity_a | wild_type.entity_b)
                if hasattr(wild_type, "node_feature"):
                    with wild_type.node():
                        wild_type.node_feature = wild_type.node_feature.to_sparse()

                if _mutant not in data_dict: continue
                mutant = data_dict[_mutant][0]
                with mutant.node():
                    entity_a = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    for a in chain_a:
                        entity_a |= mutant.chain_id == mutant.alphabet2id[a]
                    mutant.entity_a = entity_a[mutant.atom2residue]

                    entity_b = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    for b in chain_b:
                        entity_b |= mutant.chain_id == mutant.alphabet2id[b]
                    mutant.entity_b = entity_b[mutant.atom2residue]

                    is_mutation = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    for m in mutations:
                        if m[-2].isalpha():
                            is_mutation |= \
                                (mutant.chain_id == mutant.alphabet2id[m[1]]) & \
                                (mutant.residue_number == int(m[2:-2])) & \
                                (mutant.insertion_code == mutant.alphabet2id[m[-2]])
                        else:
                            is_mutation |= \
                                (mutant.chain_id == mutant.alphabet2id[m[1]]) & \
                                (mutant.residue_number == int(m[2:-1]))
                    mutant.is_mutation = is_mutation[mutant.atom2residue]
                mutant = mutant.subgraph(mutant.entity_a | mutant.entity_b)
                if hasattr(mutant, "node_feature"):
                    with mutant.node():
                        mutant.node_feature = mutant.node_feature.to_sparse()

                self.data.append((wild_type, mutant, float(ddG), mutations, fname.split(".")[0]))
                self.pdb_files.append(data_dict[_mutant][1])
                self.sequences.append((data_dict[_wild_type][2], data_dict[_mutant][2]))

    def split(self, test_set="split_0", valid_ratio=0.1):
        indices = list(range(len(self)))
        train_indices = []
        offset = 0
        for split, num_samples in zip(self.splits, self.num_samples):
            if split != test_set:
                train_indices += indices[offset: offset + num_samples]
            offset += num_samples

        idx = self.splits.index(test_set)
        num_samples = self.num_samples[idx]
        offset = sum(self.num_samples[:idx])
        test_indices = indices[offset: offset + num_samples]

        num_val_samples = int(len(train_indices) * valid_ratio)
        valid_indices = np.random.choice(train_indices, num_val_samples, replace=False)
        train_indices = [idx for idx in train_indices if idx not in valid_indices]

        return [
            torch_data.Subset(self, train_indices),
            torch_data.Subset(self, valid_indices),
            torch_data.Subset(self, test_indices)
        ]

    def get_item(self, index):
        if getattr(self, "lazy", False):
            mutant = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
            wild_type = data.Protein.from_pdb(
                os.path.join(os.path.dirname(self.pdb_files[index]), "WT_" + os.path.basename(self.pdb_files[index])),
                self.kwargs
            )
        else:
            wild_type = self.data[index][0].clone()
            mutant = self.data[index][1].clone()

        wt_residue_feature = F.one_hot(wild_type.residue_type, len(data.Protein.residue2id)+1)
        # wt_atom_feature = F.one_hot(wild_type.atom_name, len(data.Protein.atom_name2id)+1)
        wt_atom_feature = torch.cat([
            F.one_hot(wild_type.atom_name, residue_constants.atom_type_num),
            wt_residue_feature[wild_type.atom2residue]
        ], dim=-1)
        with wild_type.node():
            wild_type.node_feature = wt_atom_feature
        with wild_type.residue():
            wild_type.residue_feature = wt_residue_feature

        mt_residue_feature = F.one_hot(mutant.residue_type, len(data.Protein.residue2id)+1)
        # mt_atom_feature = F.one_hot(mutant.atom_name, len(data.Protein.atom_name2id)+1)
        mt_atom_feature = torch.cat([
            F.one_hot(mutant.atom_name, residue_constants.atom_type_num),
            mt_residue_feature[mutant.atom2residue]
        ], dim=-1)
        with mutant.node():
            mutant.node_feature = mt_atom_feature
        with mutant.residue():
            mutant.residue_feature = mt_residue_feature
        # if hasattr(wild_type, "node_feature"):
        #     with wild_type.node():
        #         wild_type.node_feature = wild_type.node_feature.to_dense()
        # if hasattr(wild_type, "residue_feature"):
        #     with wild_type.residue():
        #         wild_type.residue_feature = wild_type.residue_feature.to_dense()
        # if hasattr(mutant, "node_feature"):
        #     with mutant.node():
        #         mutant.node_feature = mutant.node_feature.to_dense()
        # if hasattr(mutant, "residue_feature"):
        #     with mutant.residue():
        #         mutant.residue_feature = mutant.residue_feature.to_dense()
        item = {"wild_type": wild_type, "mutant": mutant}
        if self.transform:
            item = self.transform(item)
        item["ddG"] = self.data[index][2]
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: ddG",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


@R.register("datasets.HER2")
class HER2(SKEMPI):

    processed_file = "1n8z_0328.pkl.gz"
    splits = ["1n8z_renum.pdb_HL_C"]

    def __init__(self, path, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            csv_files = []
            for split in self.splits:
                split_path = os.path.join(path, split)
                pdb_files += sorted(glob.glob(os.path.join(split_path, "data", "*.pdb")))
                csv_files.append(os.path.join(split_path, "data.csv"))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.load_annotation(csv_files)
            pdb_files = []
            for pdb_file in self.pdb_files:
                pdb_dir, pdb_name = os.path.split(pdb_file)
                split = os.path.basename(os.path.dirname(pdb_dir))
                pdb_file = os.path.join(split, pdb_name)
                pdb_files.append(pdb_file)
            self.pdb_files = pdb_files
            self.save_pickle(pkl_file, verbose=verbose)

        pdb_splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [pdb_splits.count(split) for split in self.splits]

    def split(self, test_set="1n8z_renum.pdb_HL_C"):
        indices = list(range(len(self)))
        test_indices = []
        offset = 0
        for split, num_samples in zip(self.splits, self.num_samples):
            if split == test_set:
                test_indices += indices[offset: offset + num_samples]
            offset += num_samples

        return [
            torch_data.Subset(self, test_indices),
            torch_data.Subset(self, test_indices),
            torch_data.Subset(self, test_indices)
        ]


@R.register("datasets.CR3022")
class CR3022(HER2):

    processed_file = "CR3022.pkl.gz"
    splits = ["6xc3_ba11_renum.pdb_C_HL", "6xc3_ba4_renum.pdb_C_HL", "6xc3_wt_renum.pdb_C_HL"]


atom_type_mapping = torch.tensor([data.feature.atom_vocab[n[0]] for n in residue_constants.atom_order])     # (37, )
atom_name_mapping = torch.tensor([data.Protein.atom_name2id[n] for n in residue_constants.atom_order])      # (37, )
inv_atom_name_mapping = torch.zeros((len(data.Protein.atom_name2id)), dtype=torch.long)
inv_atom_name_mapping[atom_name_mapping] = torch.arange(residue_constants.atom_type_num, dtype=torch.long)      # (37, )
residue_type_mapping = torch.tensor([data.Protein.residue_symbol2id.get(n, 0) for n in residue_constants.restypes_with_x])    # (21, )


def load_protein(data_dict):
    atom_mask = torch.tensor(data_dict['atom_mask']).bool()
    atom_type = atom_type_mapping[None, :]
    atom_type = atom_type.expand_as(atom_mask)[atom_mask]
    atom_name = atom_name_mapping[None, :]
    atom_name = atom_name.expand_as(atom_mask)[atom_mask]
    node_position = torch.tensor(data_dict['atom_positions'])[atom_mask]
    residue_type = torch.tensor(data_dict['aatype'])
    residue_type = residue_type_mapping[residue_type]
    residue_number = torch.tensor(data_dict['residue_index'])
    b_factor = torch.tensor(data_dict['b_factors'])[atom_mask]
    chain_id = torch.tensor(data_dict['chain_index'])
    num_residue = residue_type.shape[0]
    num_atom = atom_name.shape[0]

    atom2residue = torch.arange(num_residue)[:, None]
    atom2residue = atom2residue.expand_as(atom_mask)[atom_mask]

    edge_list = torch.zeros((1, 3), dtype=torch.long)
    bond_type = torch.zeros((1,), dtype=torch.long)

    residue_feature = F.one_hot(residue_type, len(residue_constants.restypes_with_x))
    atom_feature = torch.cat([
        F.one_hot(atom_name, residue_constants.atom_type_num),
        residue_feature[atom2residue]
    ], dim=-1)

    protein = data.Protein(edge_list=edge_list, atom_type=atom_type, bond_type=bond_type,
                        residue_type=residue_type, atom_name=atom_name, atom2residue=atom2residue,
                        residue_feature=residue_feature, atom_feature=atom_feature, bond_feature=None,
                        residue_number=residue_number, b_factor=b_factor, chain_id=chain_id,
                        node_position=node_position, num_node=num_atom, num_residue=num_residue,
    )
    return protein
