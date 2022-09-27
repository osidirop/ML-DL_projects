# Anomaly detection on network traffic from Internet of Things (IoT) devices

### Introduction

> Internet of Things (IoT) devices are used in many different sectors, for industrial and infrastucture applications and for consumer and commercial use and are rapidly growing the last years. This technology provides a great level of automation and information sharing to enhance usability and functionality <a id = "ref_1">[1]</a>. However, the number of threats and cyber-attacks affecting IoT networks is also constantly growing. This posses concerns related to network security and data privacy, as IoT devices usually lack of proper control measures and proactive security management
(e.g. usage of default passwords, no firmware updates, no access control policy) <a id = "ref_2">[2]</a>, <a id = "ref_3">[3]</a>. 

> An IoT device, without proper security, can easily become a botnet <a id = "ref_4">[4]</a>, giving for e.g. remote access and control over the Internet to a botmaster, using a Command&Control (C&C) server. Botnets have been used to perpetrate a wide range of malicious attacks, from massive SPAM and phishing campaigns to distributed denial-of-service (DDoS) <a id = "ref_5">[5]</a>. A DDoS attack saturates the targeted server or network with more traffic than it can handle provoking the service or website to crash and become unavailable to legitimate users requests.

> Using a publicly available dataset of network traffic from IoT devices, the aim of this project is to detect the different type of flows (benign or malicious) in network activities.


<br>

> <u>References</u>:

> [[1]](#ref_1) https://en.wikipedia.org/wiki/Internet_of_things

> [[2]](#ref_2) W. Zhou, Y. Jia, A. Peng, Y. Zhang and P. Liu, "The Effect of IoT New Features on Security and Privacy: New Threats, Existing Solutions, and Challenges Yet to Be Solved," in IEEE Internet of Things Journal, vol. 6, no. 2, pp. 1606-1616, April 2019, doi: https://doi.org/10.1109/JIOT.2018.2847733.

> [[3]](#ref_3) A. Hamza, H. H. Gharakheili, V. Sivaraman, "IoT Network Security: Requirements, Threats, and Countermeasures", August 2020, doi:
https://doi.org/10.48550/arXiv.2008.09339

> [[4]](#ref_4) https://en.wikipedia.org/wiki/Botnet

> [[5]](#ref_5) https://en.wikipedia.org/wiki/Denial-of-service_attack


### The data

#### a) Overview

> The dataset, called **IoT-23**, is a labeled dataset with malicious and benign IoT network traffic <a id = "ref_1">[1]</a> and was created by the Stratosphere Research Laboratory at the CTU University in Czech Republic <a id = "ref_2">[2]</a>. 
It contains 20 malware captures (called scenarios) and 3 captures for benign IoT traffic. For each malicious scenario, a specific malware sample that was running on a Raspberry Pi, used several protocols and performed different actions. The benign scenarios were obtained from three different IoT devices:
>  - a smart LED lamp (Philips HUE), 
>  - a home intelligent personal assistant (Amazon Echo) and 
>  - a smart door lock (Somfy)

> Both malicious and benign scenarios run in a controlled network environment with unrestrained internet connection like any other real IoT device <a id = "ref_2">[2]</a>.

> There are two types of datasets available:
>  - The full IoT-23 dataset which is 21 GB and contains the packet captures (pcap) files 
>  - A lighter version without the pcap files which is 8.8 GB 

> For the purpose of this project the lighter version is used. The available log files contain information from the open source network security monitoring tool called Zeek <a id = "ref_3">[3]</a> which was used to analyse the pcap files providing some additional information as for e.g. the IP addresses, ports, bytes or packets transmitted. Each entry was manually labeled by the Stratosphere Research Laboratory for **binary and/or multi-class classification scenarios**. 

<br>

> <u>References</u>:

> [[1]](#ref_1) Sebastian Garcia, Agustin Parmisano, & Maria Jose Erquiaga. (2020). IoT-23: A labeled dataset with malicious and benign IoT network traffic (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4743746

> [[2]](#ref_2) https://www.stratosphereips.org/datasets-iot23

> [[3]](#ref_3) https://zeek.org

#### b) Processing

> Most of the available log files are quite large (up to several GBs) and cannot be processed neither locally nor in google colabs. The *Modin library* which can load many GBs was also tried but it conflicts with the pandas library. The log files that can be succesfully proccessed locally are up to ~24.5 MB. They are the 3 benign and 7 malware captures:

> 1. CTU-Honeypot-Capture-4-1 (data from the smart LED lamp)
> 2. CTU-Honeypot-Capture-5-1 (data from the home intelligent personal assistant)
> 3. CTU-Honeypot-Capture-7-1 (data from the smart door lock)
> 4. CTU-IoT-Malware-Capture-3-1 (data from Muhstik botnet)
> 5. CTU-IoT-Malware-Capture-8-1 (data from Hakai DDoS botnet)
> 6. CTU-IoT-Malware-Capture-20-1 (data from Torii Botnet)
> 7. CTU-IoT-Malware-Capture-21-1 (data from Torii Botnet)
> 8. CTU-IoT-Malware-Capture-34-1 (data from Mirai Botnet)
> 9. CTU-IoT-Malware-Capture-42-1 (data from Trojan malware which downloads onto a computer programs disguised as legitimate)
> 10. CTU-IoT-Malware-Capture-44-1 (data from Mirai Botnet)

> Each log file contains also some unuseful information that needs to be dropped. For this reason, after retreiving all the 10 absolute paths of the log files, each one is read with the pandas *read_table* function by skiping the unnecessary lines. The resulting 10 dataframes, are then merged into a single one and saved into a csv file for easily access through the different notebooks. 

### Notebooks
There are 7 notebooks in total numbered. Their content is the following:
> 01. The log files are read from the data directory and merged. A new csv file ~25MB is produced. This file is then the one read by the second notebook.
> 02. Contains the Exploratory Data Analysis
> 03. Contains tests performed with a classifier from the observations obtained in the EDA in order to decide the final preprocessing steps of the data. 
> 04. Since the final features are more than 100, the PCA and the SeleckBest are explored in order to reduce then number of features.
> 05. Contains tests with 7 mutliclass classifiers without fine tuning. 
> 06. Contains fine tunning of 2 models selected by the previous tests. The models are the Logistic Regression ovr and the SVC with rbf kernel. In this notebook those 2 models are tuned for their regularization strength. After fine tuning, the classfiers are tested on the test set.
> 07. The results obtained from all classifiers are presented and discussed. 
