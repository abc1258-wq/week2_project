# PROJECT LOG – WEEK 2 (DATA ANALYTICS)

## Project Title
AI-Powered Threat Detection in Cybersecurity: Overcoming Limitations of Traditional Systems

## Week
Week 2

## Focus
Data Ingestion, Cleaning Plan, ETL Staging, and Initial Exploratory Data Analysis

---

## 1. Summary of Work Completed

During Week 2, data analytics activities have been completed to prepare network traffic data for subsequent analytical and modelling stages. A cybersecurity-related dataset containing network connection records has been ingested and validated. A structured cleaning plan has been defined based on data quality checks. An Extract–Transform–Load (ETL) pipeline has been implemented to convert raw data into an analysis-ready format. Initial exploratory data analysis has been performed to understand traffic distribution, feature relationships, and potential data quality risks. This work builds directly on the Week 1 literature review by operationalising the data foundation required for AI-based intrusion detection.

---

## 2. Evidence of Technical Progress

The following technical progress has been achieved and documented:

- Successful dataset import and schema validation  
- Verification of dataset dimensions and feature structure  
- Data quality checks including missing values and duplicate detection  
- Execution of ETL pipeline with categorical feature encoding  
- Generation of processed dataset for reuse  
- Creation of multiple exploratory data visualisations  
- Computation of summary and group-wise statistics  

Supporting evidence includes screenshots of terminal output, generated plots, and processed files.

---

## 3. Data Ingestion Notes

**Datasets imported:**  
- NSL-KDD training dataset (KDDTrain.txt), containing labelled network traffic records

**Issues found:**  
- No missing values detected  
- No duplicate records detected  

**Data validation comments:**  
- Feature names and data types have been validated  
- Dataset size and structure are consistent with documented specifications  

---

## 4. Cleaning Plan Summary

The following cleaning and preparation strategy has been defined:

- **Missing values:** None detected; no imputation required  
- **Duplicates:** Duplicate checks performed; no duplicates identified  
- **Data types:** Numeric and categorical features validated  
- **Outliers:** Presence of extreme values identified using boxplots; retained for analytical relevance  
- **Formatting:** Categorical features scheduled for encoding during ETL  

This plan ensures data integrity while preserving behavioural characteristics relevant to intrusion detection.

---

## 5. ETL Staging Details

**Extract:**  
- Raw network traffic data loaded from text file into the analysis environment  

**Transform:**  
- Binary traffic classification created (Normal vs Attack)  
- Categorical attributes encoded using one-hot encoding  
- Feature structure standardised for analytical processing  

**Load:**  
- Transformed dataset exported as `week2_processed_data.csv` for downstream use  

---

## 6. Initial Exploratory Data Analysis Findings

Initial exploratory analysis has revealed several important insights:

- Class distribution analysis shows both normal and attack traffic are well represented, with slight imbalance  
- Source byte distributions exhibit significant variability and extreme values, particularly for attack traffic  
- Correlation analysis highlights strong relationships among connection count, service rate, and error rate features  
- Summary statistics confirm higher dispersion and variance within attack-related records  

These findings support the suitability of the dataset for advanced analytical and AI-based intrusion detection methods.

---

## 7. Issues and Risks Identified

The following risks have been identified during Week 2:

- Class imbalance may introduce bias during modelling  
- Highly correlated features may affect model stability  
- Extreme values may influence distance-based or statistical methods  

Mitigation strategies will be considered during feature engineering and model selection in later stages.

---

## 8. Plan for Week 3

Planned activities for Week 3 include:

- Feature engineering and selection  
- Implementation of baseline analytical or detection models  
- Evaluation of traditional detection performance  
- Comparison groundwork for AI-based detection approaches  
- Continued documentation and validation  

---

## Status

Week 2 objectives have been successfully completed.  
The dataset is validated, transformed, and ready for analytical modelling.
