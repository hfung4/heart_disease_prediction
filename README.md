## Heart disease prediction

- Use clinical and demographic information of 300 patients to predict whether or not they have heart disease

- Data comes from the UCI database (Cleveland)

- 15 features (some are derived) were used to train model

- An optimized logistic regression model had a performance of test roc_auc >0.9 

- However, data is not representative of the Cleveland population in 2020

- **Main insights**:
  
  - The most important predictors are clinical data that are obtainable from pathology labs and health care providers. This suggest that we can partner with these entities to obtain more data. Demographics also plays a role as male is substantially more likely to have heart disease than their female counterparts. Additional demographics characteristics could be collected from patients.
    - Worryingly, patients with asymptomatic chest pain (absence of classical acute coronary syndrome (ACS) pain) are 4.5 times more likely to have heart disease than those with typical angina chest pain. Beacuse it's not immediately noticebale, heart disease patients with asymptomatic chest pains may face delayed diagnosis and treatment.
    
    - `exang` (exercise induced angina) is a metric that is only measureable at the clinic: patients with exercise induced angina are 1.48 times more likely to have heart disease than those without.
    
    - thal stress tests is also predictive of heart disease: those with normal test results are only a third as likely to have heart disease than those with abnormal results.
    
    - Patients with upsloping ST segment are half as likely to have heart disease than those with downsloping ST segment.

- **Next steps:** With a larger dataset, the model can be used to generate mean predicted probabilities of heart disease in a geographic area (e.g., HSA or census tract). This allow us to gauge the demand for Pfizer's products such as Vyndaqel and Vyndamax in specific geographic areas, which in turn inform the company's production and distribution plans, as well as marketing strategies at a more precise level.
