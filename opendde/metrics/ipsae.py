"""
ipSAE / pDockQ / pDockQ2 / LIS computation (torch-native).

Adapted from Dunbrack's preprint (https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2)


This version operates directly on in-memory tensors produced by the prediction pipeline,
Adapted from evaluation engine
no CIF file parsing required, all computation in torch, GPU-friendly:
  - biotite AtomArray (for chain/residue topology)
  - PAE matrix      (N_tokens, N_tokens)
  - per-atom pLDDT  (N_atoms,)
  - atom_to_token_idx (N_atoms,)
  - predicted coords (N_atoms, 3)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union, List, Any
import re
import numpy as np
import torch
from biotite.structure import AtomArray, get_residue_starts


NUC_RESIDUE_SET = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

def _match_chain_mask(chains_np: np.ndarray, chain_spec: str) -> np.ndarray:
    """Match exact `chain_spec` or `<chain_spec><digits>` (e.g. T, T0, T1, ...)."""
    pattern = re.compile(rf"^{re.escape(chain_spec)}\d*$")
    return np.array([bool(pattern.match(str(c))) for c in chains_np], dtype=bool)

def _as_tensor(x, dtype=None, device=None) -> torch.Tensor:
    """Cast numpy/tensor input to torch.Tensor (handles bfloat16 -> float32)."""
    if isinstance(x, torch.Tensor):
        t = x
        # if t.dtype == torch.bfloat16:
        #     t = t.to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x))
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


class IPSAECalculator:
    """
    Calculates ipSAE, pDockQ, pDockQ2 and LIS metrics for a chain pair,
    operating directly on biotite AtomArray + PAE/pLDDT torch tensors.
    All math is performed in torch on the input tensors' device.
    """

    def __init__(
        self,
        pae_cutoff: float = 8.0,
        dist_cutoff: float = 8.0,
        pdockq_cutoff: float = 8.0,
    ):
        self.pae_cutoff = pae_cutoff
        self.dist_cutoff = dist_cutoff
        self.pdockq_cutoff = pdockq_cutoff

    # ------------------------------------------------------------------ #
    # d0 / pTM helpers (torch)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calc_d0_scalar(L: Union[float, int], pair_type: str = "protein") -> float:
        """Scalar d0 — returned as a python float (no autograd needed)."""
        L = float(L)
        min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
        if L > 27:
            d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
        else:
            d0 = 1.0
        return max(min_value, d0)

    @staticmethod
    def _calc_d0_tensor(L: torch.Tensor, pair_type: str = "protein") -> torch.Tensor:
        """Vectorized d0 in torch."""
        L = L.to(torch.float32)
        L = torch.clamp(L, min=26.0)
        min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
        d0 = 1.24 * (L - 15.0).pow(1.0 / 3.0) - 1.8
        return torch.clamp(d0, min=min_value)

    @staticmethod
    def _ptm_func(x: torch.Tensor, d0: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(d0, torch.Tensor):
            d0 = torch.as_tensor(d0, dtype=x.dtype, device=x.device)
        return 1.0 / (1.0 + (x / d0) ** 2.0)

    # ------------------------------------------------------------------ #
    # Build per-residue CA table from AtomArray (CPU/python, runs once)
    # ------------------------------------------------------------------ #
    def _residue_table(self, atom_array: AtomArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each polymer residue, find its CA (or C1') atom.
        Returns:
            ca_atom_idx: (n_res,) int64 numpy array of atom indices in AtomArray
            chains:      (n_res,) object numpy array of chain ids (str)
        """
        res_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        ca_atom_idx = []
        chains_list = []

        for r in range(len(res_starts) - 1):
            i0, i1 = res_starts[r], res_starts[r + 1]
            atom_names = atom_array.atom_name[i0:i1]
            res_name = str(atom_array.res_name[i0])

            rep_local = None
            if res_name in NUC_RESIDUE_SET:
                for k, name in enumerate(atom_names):
                    if str(name).startswith("C1"):
                        rep_local = k
                        break
            else:
                for k, name in enumerate(atom_names):
                    if str(name) == "CA":
                        rep_local = k
                        break
            if rep_local is None:
                continue  # ligand / ion / unrecognized

            ca_atom_idx.append(int(i0 + rep_local))
            chains_list.append(str(atom_array.chain_id[i0]))

        return np.array(ca_atom_idx, dtype=np.int64), np.array(chains_list)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute(
        self,
        atom_array: AtomArray,
        pae_matrix: Union[np.ndarray, torch.Tensor],
        plddt_vector: Union[np.ndarray, torch.Tensor],
        atom_to_token_idx: Union[np.ndarray, torch.Tensor],
        pred_coordinates: Union[np.ndarray, torch.Tensor, None] = None,
        binder_chain: str = "H",
        target_chain: str = "T",
    ) -> Dict[str, float]:
        """
        Computes the ipSAE metrics for a single sample (e.g. prediction/pred_dict['full_data'][i])
        In a way this is similar to `protenix.model.sample_confidence._compute_full_data_and_summary`
        Args:
            atom_array:        biotite AtomArray for the predicted complex.
            pae_matrix:        (N_tokens, N_tokens) PAE matrix (torch or numpy).
            plddt_vector:      (N_atoms,) per-atom pLDDT, 0-1 or 0-100.
            atom_to_token_idx: (N_atoms,) mapping atom -> token index in PAE.
            pred_coordinates:  optional (N_atoms, 3) predicted coords.
                               If None, uses atom_array.coord.
            binder_chain / target_chain: chain ids.

        Returns:
            dict of python floats (ready to be JSON-serialized).
        """
        try:
            # ----- pick device & dtype from PAE -----
            pae = pae_matrix
            if not isinstance(pae, torch.Tensor):
                pae = torch.as_tensor(np.asarray(pae))
            if pae.dtype == torch.bfloat16:
                pae = pae.to(torch.float32)
            if pae.ndim == 3 and pae.shape[0] == 1:
                pae = pae[0]

            # Test: Pass everything to CPU due to CUDA memory constraints
            device = 'cpu'
            pae = pae.to(torch.float32).to(device)
            # device = pae.device
            dtype = pae.dtype

            plddt = _as_tensor(plddt_vector, dtype=dtype, device=device)
            atom_to_token = _as_tensor(atom_to_token_idx, dtype=torch.long, device=device)

            # pLDDT scaling
            if plddt.numel() and torch.nanquantile(plddt, 1.0) <= 1.0:
                plddt = plddt * 100.0

            # coords
            if pred_coordinates is not None:
                coords_all = _as_tensor(pred_coordinates, dtype=dtype, device=device)
            else:
                coords_all = _as_tensor(atom_array.coord, dtype=dtype, device=device)

            # ----- per-residue CA table (numpy, then push to device) -----
            ca_idx_np, chains_np = self._residue_table(atom_array)
            numres = len(ca_idx_np)
            if numres == 0:
                return self._empty_result()

            ca_atom_idx = torch.as_tensor(ca_idx_np, dtype=torch.long, device=device)

            # residue-level views
            ca_token_idx = atom_to_token[ca_atom_idx]                       # (numres,)
            pae_res = pae.index_select(0, ca_token_idx).index_select(1, ca_token_idx)
            plddt_res = plddt[ca_atom_idx]                                  # (numres,)
            coords = coords_all[ca_atom_idx]                                # (numres, 3)

            # distance matrix between CAs
            distances = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)

            # ----- chain index tensors -----
            c1_mask_np = _match_chain_mask(chains_np, target_chain)
            c2_mask_np = (chains_np == binder_chain)
            if not c1_mask_np.any() or not c2_mask_np.any():
                return self._empty_result()

            c1_indices = torch.as_tensor(np.where(c1_mask_np)[0], dtype=torch.long, device=device)
            c2_indices = torch.as_tensor(np.where(c2_mask_np)[0], dtype=torch.long, device=device)

            # ----- pDockQ / pDockQ2 -----
            pdockq_t = torch.zeros((), dtype=dtype, device=device)
            pdockq2_t = torch.zeros((), dtype=dtype, device=device)
            mean_plddt_t = torch.zeros((), dtype=dtype, device=device)

            contact_mask = distances <= self.pdockq_cutoff               # (numres, numres)
            interface_mask = contact_mask.index_select(0, c1_indices).index_select(1, c2_indices)
            npairs_t = interface_mask.sum()
            npairs = int(npairs_t.item())

            if npairs > 0:
                c1_in_if = interface_mask.any(dim=1)                     # (|c1|,)
                c2_in_if = interface_mask.any(dim=0)                     # (|c2|,)
                c1_if_idx = c1_indices[c1_in_if]
                c2_if_idx = c2_indices[c2_in_if]

                interface_plddts = torch.cat([plddt_res[c1_if_idx], plddt_res[c2_if_idx]])
                mean_plddt_t = interface_plddts.mean()

                x_q1 = mean_plddt_t * math.log10(npairs)
                pdockq_t = 0.724 / (1.0 + torch.exp(-0.052 * (x_q1 - 152.611))) + 0.018

                pae_c1c2 = pae_res.index_select(0, c1_indices).index_select(1, c2_indices)
                pae_contacts = pae_c1c2[interface_mask]
                mean_ptm = self._ptm_func(pae_contacts, 10.0).mean()
                x_q2 = mean_plddt_t * mean_ptm
                pdockq2_t = 1.31 / (1.0 + torch.exp(-0.075 * (x_q2 - 84.733))) + 0.005

            # ----- LIS -----
            sub_pae = pae_res.index_select(0, c1_indices).index_select(1, c2_indices)
            valid_pae = sub_pae[sub_pae <= 12.0]
            if valid_pae.numel() > 0:
                lis_t = ((12.0 - valid_pae) / 12.0).mean()
            else:
                lis_t = torch.zeros((), dtype=dtype, device=device)

            # ----- ipSAE (directional, vectorized) -----
            def calc_directional_scores(idxA: torch.Tensor, idxB: torch.Tensor):
                sub_pae_ab = pae_res.index_select(0, idxA).index_select(1, idxB)  # (|A|, |B|)
                valid_mask = sub_pae_ab < self.pae_cutoff                          # (|A|, |B|)

                n0chn = idxA.numel() + idxB.numel()
                d0chn = self._calc_d0_scalar(n0chn)

                n0dom = int(valid_mask.any(dim=1).sum().item()) + int(valid_mask.any(dim=0).sum().item())
                d0dom = self._calc_d0_scalar(n0dom)

                n0res = valid_mask.sum(dim=1)                                      # (|A|,)
                d0res = self._calc_d0_tensor(n0res)                                # (|A|,)

                # Compute ptm matrices for the three d0 variants.
                # chn / dom are scalar d0 → straightforward
                ptm_chn = self._ptm_func(sub_pae_ab, d0chn)                        # (|A|, |B|)
                ptm_dom = self._ptm_func(sub_pae_ab, d0dom)                        # (|A|, |B|)
                # res has a per-row d0 → broadcast d0 over columns
                ptm_res = 1.0 / (1.0 + (sub_pae_ab / d0res.unsqueeze(1)) ** 2.0)   # (|A|, |B|)

                # Mean over valid columns per row.
                # Use masked sum / count to keep it vectorized.
                valid_f = valid_mask.to(dtype)
                counts = valid_f.sum(dim=1).clamp_min(1.0)                         # (|A|,)
                has_any = n0res > 0                                                # (|A|,)

                row_mean_chn = (ptm_chn * valid_f).sum(dim=1) / counts
                row_mean_dom = (ptm_dom * valid_f).sum(dim=1) / counts
                row_mean_res = (ptm_res * valid_f).sum(dim=1) / counts

                row_mean_chn = torch.where(has_any, row_mean_chn, torch.zeros_like(row_mean_chn))
                row_mean_dom = torch.where(has_any, row_mean_dom, torch.zeros_like(row_mean_dom))
                row_mean_res = torch.where(has_any, row_mean_res, torch.zeros_like(row_mean_res))

                return {
                    "chn": row_mean_chn.max() if row_mean_chn.numel() else torch.zeros((), dtype=dtype, device=device),
                    "dom": row_mean_dom.max() if row_mean_dom.numel() else torch.zeros((), dtype=dtype, device=device),
                    "res": row_mean_res.max() if row_mean_res.numel() else torch.zeros((), dtype=dtype, device=device),
                }

            res_AB = calc_directional_scores(c1_indices, c2_indices)
            res_BA = calc_directional_scores(c2_indices, c1_indices)

            def _mx(a, b): return torch.maximum(a, b)
            def _mn(a, b): return torch.minimum(a, b)

            # ----- pack results (single GPU->CPU sync here) -----
            return {
                "ipsae_local_max":      _mx(res_AB["res"], res_BA["res"]).item(),
                "ipsae_global_max":     _mx(res_AB["chn"], res_BA["chn"]).item(),
                "ipsae_interface_max":  _mx(res_AB["dom"], res_BA["dom"]).item(),
                "iptm_global_max":      _mx(res_AB["chn"], res_BA["chn"]).item(),
                "ipsae_local_min":      _mn(res_AB["res"], res_BA["res"]).item(),
                "ipsae_global_min":     _mn(res_AB["chn"], res_BA["chn"]).item(),
                "ipsae_interface_min":  _mn(res_AB["dom"], res_BA["dom"]).item(),
                "iptm_global_min":      _mn(res_AB["chn"], res_BA["chn"]).item(),
                "pdockq":               float(pdockq_t.item()),
                "pdockq2":              float(pdockq2_t.item()),
                "lis":                  float(lis_t.item()),
                "npairs_pdockq":        npairs,
                "mean_plddt_interface": float(mean_plddt_t.item()),
            }

        except Exception as e:
            print(f"  [ERROR] Failed to compute IPSAE metrics: {e}")
            return self._empty_result()

    def compute_update_confidence(self, atom_array, pred_dict, binder_chain="H", target_chain="T") -> List[Dict]:
        """Computes ipSAE results and updates the sample confidence, one per sample."""
        for _pdict, conf in zip(pred_dict['full_data'], pred_dict['summary_confidence']):
            res = self._compute(atom_array, pae_matrix=_pdict['token_pair_pae'],
                                plddt_vector=_pdict['atom_plddt'],
                                atom_to_token_idx=_pdict['atom_to_token_idx'],
                                pred_coordinates= _pdict['atom_coordinate'],
                                binder_chain=binder_chain, target_chain=target_chain)
            conf.update(res)


    @staticmethod
    def _empty_result() -> Dict[str, float]:
        return {
            "ipsae_local_max": 0.0,
            "ipsae_global_max": 0.0,
            "ipsae_interface_max": 0.0,
            "iptm_global_max": 0.0,
            "ipsae_local_min": 0.0,
            "ipsae_global_min": 0.0,
            "ipsae_interface_min": 0.0,
            "iptm_global_min": 0.0,
            "pdockq": 0.0,
            "pdockq2": 0.0,
            "lis": 0.0,
            "npairs_pdockq": 0,
            "mean_plddt_interface": 0.0,
        }