"""AutoDock Vina docking wrapper for batch scoring of generated molecules."""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


class VinaDocker:
    def __init__(
        self,
        receptor_pdbqt: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
        exhaustiveness: int = 8,
        num_modes: int = 9,
        vina_bin: str = "vina",
    ):
        self.receptor = receptor_pdbqt
        self.center = center
        self.box_size = box_size
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.vina_bin = vina_bin

    def _mol_to_pdbqt(self, mol: Chem.Mol, tmp_dir: str) -> Optional[str]:
        """Convert RDKit Mol to PDBQT via Open Babel."""
        sdf_path = os.path.join(tmp_dir, "ligand.sdf")
        pdbqt_path = os.path.join(tmp_dir, "ligand.pdbqt")

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)

        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

        result = subprocess.run(
            ["obabel", sdf_path, "-O", pdbqt_path, "--partialcharge", "gasteiger"],
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.exists(pdbqt_path):
            return None
        return pdbqt_path

    def dock_mol(self, mol: Chem.Mol) -> Optional[float]:
        """Dock a single RDKit Mol; return best Vina score or None on failure."""
        with tempfile.TemporaryDirectory() as tmp:
            ligand_pdbqt = self._mol_to_pdbqt(mol, tmp)
            if ligand_pdbqt is None:
                return None

            out_path = os.path.join(tmp, "out.pdbqt")
            cmd = [
                self.vina_bin,
                "--receptor", self.receptor,
                "--ligand", ligand_pdbqt,
                "--center_x", str(self.center[0]),
                "--center_y", str(self.center[1]),
                "--center_z", str(self.center[2]),
                "--size_x", str(self.box_size[0]),
                "--size_y", str(self.box_size[1]),
                "--size_z", str(self.box_size[2]),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--out", out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None

            # Parse best score from Vina output
            for line in result.stdout.splitlines():
                if line.strip().startswith("1 "):
                    try:
                        return float(line.split()[1])
                    except (IndexError, ValueError):
                        pass
        return None


def dock_molecules(
    sdf_file: str,
    docker: VinaDocker,
    output_csv: str,
) -> pd.DataFrame:
    """
    Dock all molecules in an SDF file and return a DataFrame with Vina scores.
    Saves intermediate results every 50 molecules for crash recovery.
    """
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    records = []

    for i, mol in enumerate(tqdm(supplier, desc="Docking")):
        if mol is None:
            records.append({"idx": i, "smiles": None, "vina_score": None, "status": "parse_error"})
            continue

        smiles = Chem.MolToSmiles(mol)
        score = docker.dock_mol(mol)
        records.append({
            "idx": i,
            "smiles": smiles,
            "vina_score": score,
            "status": "ok" if score is not None else "docking_failed",
        })

        # Checkpoint every 50 molecules
        if (i + 1) % 50 == 0:
            pd.DataFrame(records).to_csv(output_csv + ".tmp", index=False)

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Docking results saved → {output_csv}  ({df['vina_score'].notna().sum()}/{len(df)} succeeded)")
    return df
