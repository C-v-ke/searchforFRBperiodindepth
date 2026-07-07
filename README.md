# A Second-scale Periodicity in an Active Repeating Fast Radio Burst Source

This repository contains the source code and data used to generate all the figures for the manuscript, "A second-scale periodicity in an active repeating fast radio burst source."

The data reduction and periodicity search (implemented in pure Python) are included within the .ipynb notebooks and .py files.

***

## 📊 Plotting Notebooks

If you are interested in the figures from the manuscript, please refer to the following notebooks:

* For figures in the **Main text**, please refer to `Plotting Figures.ipynb`.
* For figures in the **Supplementary Information**, please refer to `Plotting Supplementary Figures.ipynb`.

***

## 🔬 Simulated Data Generation

The following files contain the code for generating the simulated data.

* `Significance_analysis.py` is used to generate the simulated data required for plotting Figures 8,  9, 10. The steps correspond to the "Statistical significance of the periodicity" subsection.
* `Multiple_emission_sites.py` is used to generate the simulated data required for plotting Supplementary Figure 4. The steps correspond to the "Effects of multiple emitting sites" subsection.
* `Comparison_analysis.py` is used to generate the simulated data required for plotting Supplementary Figures 5 & 7. The steps correspond to the "Comparison with the results of other groups" subsection.

***

## 📁 Repository Structure

* The `Data` folder contains all data tables and the necessary data for figure generation.
* The `Periodograms` folder contains the periodicity search results for all observing days.
* The `Folded_phases` folder contains the phase distributions for all observing days.
* The `Significance_results` folder contains the simulated data of significance analysis.
* The `Comparison_results` folder contains the simulated data of comparison analysis.

