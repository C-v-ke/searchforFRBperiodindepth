# A Second-scale Periodicity in an Active Repeating Fast Radio Burst Source

This repository contains the source code and data used to generate all the figures for the manuscript, "A second-scale periodicity in an active repeating fast radio burst source."

The data reduction and periodicity search (implemented in pure Python) are included within the .ipynb notebooks.

***

## 📊 Plotting Notebooks

If you are interested in the figures from the manuscript, please refer to the following notebooks:

* For figures in the **Main text**, please refer to `Plotting Figures.ipynb`.
* For figures in the **Extended Data Figures**, please refer to `Plotting Extended Data Figures.ipynb`.

***

## 🔬 Simulated Data Generation

The following notebooks contain the steps for generating the simulated data.

* `Generate N bursts from m emission sites.ipynb` is used to generate the simulated data required for plotting **Extended Data Figure 9**. The steps correspond to the "Effects of multiple emitting sites" subsection.
* `Generate MC and BS samples.ipynb` is used to generate the simulated data required for plotting **Extended Data Figures 3 & 4**. The steps correspond to the "Statistical significance of the periodicity" subsection.

***

## 📁 Repository Structure

* The **`Data`** folder contains all data tables and the necessary data for figure generation.
* The **`Peridograms`** folder contains the periodicity search results for all observation days (**Extended Data Figure 1**).
* The **`Folded Phases`** folder contains the phase folding results for all observation days, as well as the phase folding results for three multi-day time blocks (**Extended Data Figures 7 & 8**).
