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

* `Generate MC Samples_Main.ipynb` is used to generate the simulated data required for plotting **Extended Data Figures 3 & 4**. The steps correspond to the "Statistical significance of the periodicity" subsection.
* `Generate N bursts from m emission sites.ipynb` is used to generate the simulated data required for plotting **Extended Data Figure 5**. The steps correspond to the "Effects of multiple emitting sites" subsection.
* `Generate MC Samples_Comparison.ipynb` is used to generate the simulated data required for plotting **Extended Data Figures 6 & 8**. The steps correspond to the "Comparison with the results of other groups" subsection.

***

## 📁 Repository Structure

* The **`Data`** folder contains all data tables and the necessary data for figure generation.
* The **`Periodograms`** folder contains the periodicity search results for all observation days (**Extended Data Figure 1**).
* The **`MC Samples*`** folders contain the simulated data.

