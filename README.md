# UNEASE
## Abstract: 

Static analysis tools (SATs) are widely applied to detect defects in software projects. However, SATs are overshadowed by a large number of unactionable warnings, which severely reduces the usability of SATs. To address this problem, the existing approaches commonly use machine learning (ML) techniques for actionable warning identification (AWI). For these ML-based AWI approaches, the warning feature determination is one of the most critical parts to effectively identify actionable warnings. To eliminate redundant and irrelevant warning features, ML-based AWI approaches usually incorporate feature selection to determine the feature subset by calculating the importance or correlation of features with warning labels. Nevertheless, warning labels are not always available directly in practice. Thus, it is vital and challenging to select warning features for ML-based AWI approaches when warning labels are absent.

To address the above problem, we propose an UNsupervised fEAture SElection approach called UNEASE for ML-based AWI. 
(1) UNEASE first performs the feature clustering to gather warning features into clusters, where the number of clusters is automatically determined and features in the same cluster are considered redundant.
(2) Subsequently, UNEASE performs the feature ranking to sort warning features in each cluster with three newly proposed ranking strategies and selects the top-ranked warning feature from each cluster. Based on the selected features, we train a ML classifier to identify actionable warnings.
We conduct experiments on nine large-scale and real-world warning datasets.
Compare UNEASE with seven typical feature selection techniques, the experimental results show that while taking a low cost to perform the feature selection and maintaining a low redundancy rate in the selected warning features, UNEASE still obtains the top-ranked AUC.
