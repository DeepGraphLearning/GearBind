import argparse
import subprocess
import requests
import tarfile
import tempfile
import pandas as pd
import numpy as np

from functools import partial
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

from fix_pdb import fix_pdb


MAX_DDG_STD = 1
FOLD2PDBS = [['1AO7_ABC_DE', '1B3S_A_D', '1E50_A_B', '1F5R_A_I', '1FC2_C_D', '1FR2_A_B', '1GL0_E_I', '1H9D_A_B', '1JTD_A_B', '1K8R_A_B', '1KNE_A_P', '1KTZ_A_B', '1MAH_A_F', '1MHP_HL_A', '1MLC_AB_E', '1MQ8_A_B', '1NCA_N_LH', '1QSE_ABC_DE', '1S1Q_A_B', '1SGE_E_I', '1SGY_E_I', '1TM3_E_I', '1TM7_E_I', '1TO1_E_I', '1U7F_B_AC', '1UUZ_A_D', '1VFB_AB_C', '1XGT_AB_C', '1XGU_AB_C', '1Y34_E_I', '1Y3B_E_I', '1YQV_HL_Y', '2B10_A_B', '2B11_A_B', '2BNR_ABC_DE', '2C5D_AB_CD', '2C5D_A_C', '2CCL_A_B', '2HLE_A_B', '2KSO_A_B', '2O3B_A_B', '2PYE_ABC_DE', '2QJA_AB_C', '2QJB_AB_C', '2REX_A_B', '2SGP_E_I', '2SGQ_E_I', '2UWE_ABC_EF', '2VIS_AB_C', '2VN5_A_B', '3BTF_E_I', '3BTM_E_I', '3BTW_E_I', '3EQY_A_C', '3G6D_LH_A', '3HG1_ABC_DE', '3IDX_HL_G', '3KBH_A_E', '3KUD_A_B', '3LNZ_A_B', '3N4I_A_B', '3Q3J_A_B', '3QDG_ABC_DE', '3S9D_A_B', '3SE8_HL_G', '3SZK_AB_C', '3TGK_E_I', '3U82_A_B', '3UII_A_P', '3WWN_A_B', '4BFI_A_B', '4E6K_AB_G', '4EKD_A_B', '4GNK_A_B', '4GU0_A_E', '4JEU_A_B', '4JFD_ABC_DE', '4JFF_ABC_DE', '4KRL_A_B', '4KRP_A_B', '4LRX_AB_CD', '4O27_A_B', '4P23_CD_AB', '4UYQ_A_B', '4Y61_A_B', '4YEB_A_B', '4YFD_A_B', '5E9D_AB_CDE', '5F4E_A_B', '5TAR_A_B'],
 ['1AHW_AB_C', '1B2S_A_D', '1B2U_A_D', '1C4Z_ABC_D', '1CBW_FGH_I', '1CSE_E_I', '1CSO_E_I', '1CZ8_HL_VW', '1DAN_HL_UT', '1DQJ_AB_C', '1E96_A_B', '1EAW_A_B', '1EFN_A_B', '1F47_A_B', '1FSS_A_B', '1GL1_A_I', '1KIP_AB_C', '1KIQ_AB_C', '1M9E_A_D', '1N8O_ABC_E', '1N8Z_AB_C', '1S0W_A_C', '1SMF_E_I', '1TM4_E_I', '1TM5_E_I', '1WQJ_I_B', '1X1X_A_D', '1XGR_AB_C', '1Y1K_E_I', '1YCS_A_B', '1YY9_CD_A', '2BDN_HL_A', '2BTF_A_P', '2DVW_A_B', '2G2W_A_B', '2I26_N_L', '2J0T_A_D', '2J12_A_B', '2JCC_ABC_EF', '2NOJ_A_B', '2NU0_E_I', '2OOB_A_B', '2P5E_ABC_DE', '2QJ9_AB_C', '2VIR_AB_C', '2VLQ_A_B', '3BDY_HL_V', '3BN9_B_CD', '3BT1_A_U', '3BTG_E_I', '3BTT_E_I', '3BX1_A_C', '3D3V_ABC_DE', '3D5S_A_C', '3EG5_A_B', '3EQS_A_B', '3F1S_A_B', '3HH2_AB_C', '3MZG_A_B', '3QHY_A_B', '3QIB_ABP_CD', '3SE9_HL_G', '4B0M_A_BM', '4CVW_A_C', '4FZA_A_B', '4G0N_A_B', '4G2V_A_B', '4HFK_A_BD', '4J2L_A_CD', '4JGH_ABC_D', '4K71_A_BC', '4KRO_A_B', '4L0P_A_B', '4MYW_A_B', '4NKQ_C_AB', '4OFY_A_D', '4PWX_AB_CD', '4UYP_A_D', '4WND_A_B', '5E6P_A_B', '5K39_A_B', '5M2O_A_B', '5UFE_A_B', '5UFQ_A_C'],
 ['1A4Y_A_B', '1ACB_E_I', '1BD2_ABC_DE', '1BRS_A_D', '1CT2_E_I', '1CT4_E_I', '1DVF_AB_CD', '1FY8_E_I', '1GC1_G_C', '1GRN_A_B', '1JCK_A_B', '1KBH_A_B', '1KIR_AB_C', '1LP9_ABC_EF', '1NMB_N_LH', '1OGA_ABC_DE', '1OHZ_A_B', '1P6A_A_B', '1PPF_E_I', '1TMG_E_I', '1X1W_A_D', '1XGQ_AB_C', '1Y33_E_I', '1Y3C_E_I', '1Y3D_E_I', '2A9K_A_B', '2ABZ_B_E', '2B2X_HL_A', '2B42_A_B', '2BNQ_ABC_DE', '2DSQ_I_G', '2E7L_EQ_AD', '2GYK_A_B', '2J8U_ABC_EF', '2JEL_LH_P', '2NU2_E_I', '2NU4_E_I', '2NYY_DC_A', '2NZ9_DC_A', '2OI9_AQ_BC', '2VLO_A_B', '2VLP_A_B', '2VLR_ABC_DE', '2WPT_A_B', '3AAA_AB_C', '3B4V_AB_C', '3BTE_E_I', '3BTH_E_I', '3L5X_A_HL', '3LZF_AB_HL', '3N0P_A_B', '3NCC_A_B', '3NGB_HL_G', '3NVQ_B_A', '3PWP_ABC_DE', '3QDJ_ABC_DE', '3R9A_AC_B', '3RF3_A_C', '3SE3_B_A', '3SE3_B_C', '3SEK_B_C', '3SF4_A_D', '3UIG_A_P', '4CPA_A_I', '4HRN_A_D', '4HSA_AB_C', '4NM8_ABCDEF_HL', '4P5T_CD_AB', '4U6H_AB_E', '4X4M_AB_E', '4YH7_A_B', '4ZS6_HL_A', '5C6T_HL_A'],
 ['1A22_A_B', '1AK4_A_D', '1B41_A_B', '1BJ1_HL_VW', '1CT0_E_I', '1FCC_A_C', '1JRH_LH_I', '1KAC_A_B', '1LFD_A_B', '1P69_A_B', '1QAB_ABCD_E', '1REW_AB_C', '1SBB_A_B', '1SGD_E_I', '1SGN_E_I', '1SGP_E_I', '1TM1_E_I', '1XGP_AB_C', '1XXM_A_C', '1Y4A_E_I', '2AJF_A_E', '2AW2_A_B', '2B0U_AB_C', '2B12_A_B', '2C0L_A_B', '2FTL_E_I', '2J1K_C_T', '2NY7_HL_G', '2PCB_A_B', '2VLN_A_B', '3BTD_E_I', '3D5R_A_C', '3HFM_HL_Y', '3M62_A_B', '3M63_A_B', '3MZW_A_B', '3N06_A_B', '3NCB_A_B', '3NVN_B_A', '3Q8D_A_E', '3SE4_B_A', '3SE4_B_C', '3SGB_E_I', '3VR6_ABCDEF_GH', '4GXU_ABCDEF_MN', '4JFE_ABC_DE', '4L3E_ABC_DE', '4MNQ_ABC_DE', '4OZG_ABJ_GH', '4RA0_A_C', '4RS1_A_B', '5CXB_A_B', '5CYK_A_B', '5XCO_A_B'],
 ['1BP3_A_B', '1C1Y_A_B', '1CHO_EFG_I', '1EMV_A_B', '1FFW_A_B', '1GCQ_AB_C', '1GUA_A_B', '1HE8_A_B', '1IAR_A_B', '1JTG_A_B', '1MI5_ABC_DE', '1R0R_E_I', '1SBN_E_I', '1SGQ_E_I', '1SIB_E_I', '1XD3_A_B', '1Y48_E_I', '1Z7X_W_X', '2AK4_ABC_DE', '2B0Z_A_B', '2G2U_A_B', '2GOX_A_B', '2HRK_A_B', '2NU1_E_I', '2PCC_A_B', '2SIC_E_I', '3BE1_HL_A', '3BK3_A_C', '3BP8_A_C', '3BTQ_E_I', '3C60_CD_AB', '3H9S_ABC_DE', '3LB6_A_C', '3N85_A_LH', '3NPS_A_BC', '3QFJ_ABC_DE', '3UIH_A_P', '3W2D_A_HL', '4FTV_ABC_DE', '4JPK_HL_A', '4N8V_G_ABC', '4NZW_A_B']]
PDB_TO_FIX = ['1C4Z', '2NYY', '2NZ9', '3VR6', '4GNK', '4GXU', '4K71', '4NM8']


def download_file(url, destination_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        print(f"Failed to download file. HTTP response status code: {response.status_code}")


def extract_tar_gz(file_path, destination_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=destination_path, filter="data")


def process_csv(df, max_ddg_std=MAX_DDG_STD, fold2pdbs=FOLD2PDBS):

    # parse temperature and ddG
    df['Temperature'] = df['Temperature'].apply(lambda x: float(x[:3]) if isinstance(x, str) else 298.)

    df['pKd_wt'] = -np.log10(df['Affinity_wt_parsed'])
    df['pKd_mt'] = -np.log10(df['Affinity_mut_parsed'])
    df['dG_wt'] =  (8.314/4184) * df['Temperature'] * np.log(df['Affinity_wt_parsed'])
    df['dG_mt'] =  (8.314/4184) * df['Temperature'] * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mt'] - df['dG_wt']

    def _is_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    valid = (df['Affinity_mut (M)'].apply(_is_float)) & (df['Affinity_wt (M)'].apply(_is_float))
    idf = df.loc[~valid]  # 474 invalid entries, discarded
    vdf = df.loc[valid]  # 6611 valid entries
    print(f"Total entries: {len(df)}, Entries with valid ddG: {len(vdf)}, {len(idf)} invalid entries will be discarded.")

    # filter out non-unique entries
    _vdf = vdf.set_index(['#Pdb', 'Mutation(s)_cleaned'])
    print(f'{_vdf.index.nunique()} unique #Pdb-mutation pairs detected.')  # 5747 unique #Pdb-mutation pairs

    cnt = _vdf.index.value_counts()
    not_unique_entries = cnt[cnt > 1].index  # 585 non-unique #Pdb-mutation pairs
    not_unique_df = _vdf.loc[not_unique_entries].reset_index()
    print(f"{len(not_unique_entries)} non-unique #Pdb-mutation pairs detected. This accounts for {len(not_unique_df)} entries.")

    # calculate ddG std for non-unique entries, and filter out highly-variable entries
    rows = []
    for (pdb_chains, mutations), sdf in not_unique_df.groupby(['#Pdb', 'Mutation(s)_cleaned']):
        _dict = sdf.ddG.describe().to_dict()
        _dict['#Pdb'] = pdb_chains
        _dict['Mutation(s)_cleaned'] = mutations
        rows.append(_dict)
    not_unique_stat = pd.DataFrame(rows).set_index(['#Pdb', 'Mutation(s)_cleaned'])
    bad_data = not_unique_stat.loc[(not_unique_stat['std'] > max_ddg_std)]  # 95 entries have > 0.5 ddG, 18 entries have > 1 ddG
    filtered_data = _vdf.loc[~_vdf.index.isin(bad_data.index)]
    print(f"{len(bad_data)} entries have ddG std > {max_ddg_std} and will be discarded. {len(filtered_data)} entries have ddG std <= {max_ddg_std}.")

    # aggregate data so that each #Pdb-mutation corresponds to one entry only
    dfs = []
    for (pdb_chains, mutations), sdf in filtered_data.groupby(['#Pdb', 'Mutation(s)_cleaned']):
        first_row = sdf.iloc[0].copy()
        first_row['ddG'] = sdf.ddG.mean()
        if (pdb_chains, mutations) in not_unique_stat.index:
            assert first_row['ddG'] == not_unique_stat.loc[(pdb_chains, mutations), 'mean']
        dfs.append(first_row.to_frame().T)
    aggr_data = pd.concat(dfs)
    aggr_data.index.names = ['#Pdb', 'Mutation(s)_cleaned']
    assert aggr_data.index.is_unique
    print(f"Aggregation complete. {len(aggr_data)} unique #Pdb-mutation pairs from {aggr_data.index.get_level_values('#Pdb').nunique()} #Pdbs remain in the processed dataset.")

    # split by #Pdb
    dfs = []
    for i, pdbs in enumerate(fold2pdbs):
        aggr_data.loc[pdbs, 'fold'] = i
        print(f"Fold {i}: {aggr_data.loc[pdbs].shape[0]} entries, {len(pdbs)} unique #Pdbs")

    # convert format for subsequent processing
    aggr_data['pdb_id'] = aggr_data['#Pdb'].apply(lambda x: x.split('_')[0])
    aggr_data['pdb_fname'] = aggr_data['pdb_id'] + '_Repair.pdb'
    aggr_data['chain_a'] = aggr_data['#Pdb'].apply(lambda x: x.split('_')[1])
    aggr_data['chain_b'] = aggr_data['#Pdb'].apply(lambda x: x.split('_')[2])
    aggr_data['mutation'] = aggr_data['Mutation(s)_cleaned']

    return aggr_data


def process_pdbs(pdb_dir):
    pdb_dir = Path(pdb_dir)

    # fix 1KBH. It stores all states in a single state, causing unresolvable clash.
    with open(pdb_dir / '1KBH.pdb') as f:
        lines = f.readlines()
    new_lines = lines[13471:14181] + lines[29285:]
    with open(pdb_dir / '1KBH.pdb', 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    print("Fixed 1KBH")

    # fix 4BFI
    with open(pdb_dir / '4BFI.pdb') as f:
        lines = f.readlines()
    new_lines = lines[:3062] + lines[3069:]
    with open(pdb_dir / '4BFI.pdb', 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    print("Fixed 4BFI")

    # fix PDB_TO_FIX
    for pdb_code in PDB_TO_FIX:
        fpath = pdb_dir / f'{pdb_code}.pdb'
        fix_pdb(fpath, {}, output_file=fpath)
        print(f'Fixed {pdb_code}')


def run_foldx_repair(file: Path, timeout: float = 3600, output_dir: Path = Path("repaired_pdbs")):
    command = f"foldx --command RepairPDB --ionStrength 0.15 --pdb-dir {file.parent} --output-dir {output_dir} --pdb {file.name}"
    # Use subprocess to execute the command in the shell
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        # Return a tuple of the file name and the stdout or stderr if command fails
        return (file, None) if result.returncode == 0 else (file, result.stderr)
    except subprocess.CalledProcessError as e:
        # Handle errors in the called executable
        return (file, e.stderr)
    except Exception as e:
        # Handle other exceptions such as file not found or permissions issues
        return (file, str(e).encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=Path, default=Path('SKEMPI_v2/skempi_v2.csv'), help='Path to the SKEMPI CSV file')
    parser.add_argument('--output-csv-path', type=Path, default=Path('SKEMPI_v2_processed/processed_skempi.csv'), help='Path to the processed SKEMPI CSV file')
    parser.add_argument('--pdb-dir', type=Path, default=Path('SKEMPI_v2_processed/PDBs'), help='Path to the SKEMPI PDB directory, must have name "PDBs"')
    parser.add_argument('--output-pdb-dir', type=Path, default=Path('SKEMPI_v2_processed/processed_pdbs'), help='Path to the output directory for repaired PDBs')
    parser.add_argument('--no-repair', action='store_true', help='skip FoldX RepairPDB step')
    args = parser.parse_args()


    if not args.csv_path.exists():
        args.csv_path.parent.mkdir(exist_ok=True, parents=True)
        download_file('https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv', args.csv_path)

    assert args.pdb_dir.name == 'PDBs'
    if not args.pdb_dir.exists():
        args.pdb_dir.parent.mkdir(exist_ok=True, parents=True)
        temp_fpath = Path(tempfile.gettempdir()) / 'SKEMPI2_PDBs.tgz'
        download_file('https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz', temp_fpath)
        extract_tar_gz(temp_fpath, args.pdb_dir.parent)

    df = pd.read_csv(args.csv_path, sep=';')
    aggr_data = process_csv(df)
    aggr_data.to_csv(args.output_csv_path)

    process_pdbs(args.pdb_dir)

    if args.no_repair:
        print("Skipping repair step.")
        exit()

    args.output_pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_fpaths = [p for p in args.pdb_dir.glob('*.pdb') if not p.name.startswith('.')]
    _run_foldx_repair = partial(run_foldx_repair, output_dir=args.output_pdb_dir)
    with mp.Pool(min(64, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_repair, tqdm(pdb_fpaths))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs repaired by FoldX")

    repaired_list = list(args.output_pdb_dir.glob('*.pdb'))
    assert len(repaired_list) == len(pdb_fpaths), f'Repaired pdbs ({len(repaired_list)}) != Original pdbs ({len(pdb_fpaths)})'
