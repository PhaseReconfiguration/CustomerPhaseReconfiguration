# Customer Phase Reconfiguration
This repository contains the code and data used for the **Customer Phase Reconfiguration** in electrical distribution networks. The methodology optimizes phase configurations to improve network balance and efficiency, enabling better integration of distributed energy resources (DERs) like photovoltaic panels (PVs), electric vehicles (EVs), and heat pumps (HPs).

For more detailed information, refer to the paper [here](https://orbi.uliege.be/handle/2268/327465).

## Table of Contents
- [Introduction](#introduction)
- [Required Data](#requireddata)
- [Usage](#usage)

## Introduction
This repository provides the implementation of a computationally efficient methodology for optimizing customer phase configurations in power distribution networks. The approach performs only one rephasing, optimized over a full year of DER load curves, to achieve balanced operation and minimize energy losses. The method eliminates the need for power flow computations and extensive smart meter installations, making it scalable for large-scale networks.

## Required Data
The folder \Data contains all the data required to run the methodology:
- A network (it requires a columns: ean, hh_surf, hh_size, distance).
- Different time-series for customers, PVs, EVs, and HPs.

## Usage
To run the methodology, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/PhaseReconfiguration/CustomerPhaseReconfiguration.git
   cd OptimalTPI

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run main notebook script:
   ```bash
   main.ipynb

***

For any questions or issues, please contact mvassallo@uliege.be.
