
# A Data-Driven Approach for Topology Correction in Low Voltage Networks with DERs

This repository contains the implementation code accompanying the paper:

> **"A Data-Driven Approach for Topology Correction in Low Voltage Networks with DERs"**
> *Dong Liu, Sander Timmerman, Yu Xiang, Peter Palensky, and Pedro P. Vergara*
> Delft University of Technology & Alliander N.V.
> [arXiv:2506.20238](https://arxiv.org/abs/2506.20238)

---

## 🧩 Overview

This work proposes a **data-driven framework** for identifying and correcting low-voltage distribution network (LVDN) topologies, even under **incomplete smart meter datasets**.
The method relies **solely on voltage magnitude data**, ensuring privacy preservation while improving **network observability**, **load balancing**, and **DER (Distributed Energy Resources)** management.

The framework consists of three key modules:

1. **Switch State Identification** – using supervised learning (Random Forest).
2. **User–Feeder Connection Identification** – based on hierarchical clustering with modified correlation metrics.
3. **Phase Label Identification** – identifying customer phases using correlation analysis and clustering.

Additionally, a **time-based smart meter data selection strategy** mitigates the impact of PV generation on correlation accuracy.

---

## 🧠 Method Summary

The proposed topology correction pipeline includes:

1. **Switch State Identification**

   * Encodes switch bar states using label encoding.
   * Trains a Random Forest classifier to infer switch states from voltage magnitudes.

2. **User–Feeder Connection Identification**

   * Applies modified Pearson correlation and hierarchical clustering (HC).
   * Integrates supervised (KNN/MFP-based) correction when partial labels are known.

3. **Phase Label Identification**

   * Identifies phase labels (A/B/C) using normalized voltage correlations.

4. **Time-Based Data Selection**

   * Filters time periods with low PV output (e.g., 22:00–05:00).
   * Enhances correlation reliability for phase identification.

---

## ⚙️ Dependencies

This implementation uses the following Python packages:

* `numpy`
* `pandas`
* `scikit-learn`
* `scipy`
* `matplotlib`
* `networkx`
* [`Power Grid Model (PGM)`](https://power-grid-model.readthedocs.io/en/stable/) — developed by **Alliander N.V.**

PGM is used for **power flow calculations** and **data simulation** within the case studies.

---

## 🧾 Data Description

The datasets used in this project were provided by **Alliander N.V.** and contain anonymized, real-world smart meter voltage data from low-voltage networks in the Netherlands.
Due to confidentiality agreements, these datasets **cannot be publicly released**.

If you wish to reproduce experiments, please contact Alliander for access to similar open datasets or use synthetic networks.

---

## 🧪 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/TopologyCorrection-LVDN.git
   cd TopologyCorrection-LVDN
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run example experiments:

   ```bash
   python main.py
   ```

4. Adjust configuration parameters in `config.yaml` for different case studies (e.g., number of feeders, SM availability, PV ratio).

---

## 🤝 Collaboration Acknowledgement

This research and code were developed in collaboration with **Alliander N.V.**

* The **Power Grid Model (PGM)** software used here was developed by Alliander and is publicly available as an open-source package.
* The **datasets** used in validation were provided by Alliander for research purposes only.

---

## 📄 Citation

If you use this code or methodology, please cite:

```bibtex
@article{liu2025topology,
  title={A Data-Driven Approach for Topology Correction in Low Voltage Networks with DERs},
  author={Liu, Dong and Timmerman, Sander and Xiang, Yu and Palensky, Peter and Vergara, Pedro P.},
  journal={arXiv preprint arXiv:2506.20238},
  year={2025}
}
```
