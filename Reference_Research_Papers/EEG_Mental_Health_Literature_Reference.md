# EEG and Mental-Health Literature Reference

> **Purpose:** A consolidated Markdown reference created from the three uploaded comparison-table PDFs. It preserves the information supplied in those files so it can be reused while drafting a research paper.

> **Important:** This document is a transcription/organization of the uploaded tables, not an independent verification of every paper. Where a source table says **Not specified**, uses a generic label such as “paper authors,” or omits a formal citation, that uncertainty is retained. Verify bibliographic details against the original publication before submitting a final journal manuscript.

## Source files

- **Comparison_Table_1.pdf** — 17 studies with year, authors, dataset, modality, methodology, reported accuracy/performance, identified gaps, and citations.
- **Comparison_Table_2.pdf** — 13 studies grouped into EEG Stress, EEG Depression, Multimodal, and Survey / Gap categories.
- **Comparison_Table_3.pdf** — 21 IEEE-oriented comparison entries with datasets, modalities, methods, performance, and limitations.

## How to use this file in paper writing

Use the individual study entries for the **Literature Review / Related Work** section, and use the cross-source themes near the end to help organize the **research-gap** discussion. Do not copy reported percentages across studies as if they were directly comparable: datasets, tasks, modalities, subject populations, validation procedures, and class definitions differ.

---

# Source 1 — Comparison_Table_1.pdf

### 1. W. Zhai, S. Guo (2025)
- **Dataset:** 35 participants
- **Modality:** 4-channel EEG + VR
- **Methodology / Model:** Preprocessing + Feature Extraction + RF/SVM/KNN
- **Reported performance:** 86.67%
- **Identified gap / limitation:** Lack of VR integration in existing systems; unclear channel selection and feature contribution.
- **Citation as provided in the source table:** Zhai, W. and Guo, S. (2025) ‘Anxiety status assessment method based on a portable EEG signal detection and Virtual Reality Integration System’, Proceedings of the 2025 3rd International Conference on Artificial Intelligence, Systems and Network Security, pp. 279–284. doi:10.1145/3797161.3797205.

### 2. T. Wang, F. Li, X. Zhang, L. Huang, W. Jia (2022)
- **Dataset:** 1320 EEG samples
- **Modality:** 20-channel EEG
- **Methodology / Model:** Preprocessing + 1D-CNN + LSTM comparison
- **Reported performance:** 90.53%
- **Identified gap / limitation:** Manual feature extraction in traditional ML is time-consuming and feature quality cannot be guaranteed; model and parameter optimization still needed.
- **Citation as provided in the source table:** Wang, T. et al. (2022) ‘A 1D-CNN prediction model for stroke classification based on EEG Signal’, Proceedings of the 8th International Conference on Communication and Information Processing, pp. 191–196. doi:10.1145/3571662.3571695.

### 3. C. Lloyd, L. Lemoine, R. Al-Shaikh, K. T. Ly, H. Kayan, C. Perera, N. Pham (2024)
- **Dataset:** 40 subjects
- **Modality:** 32-electrode EEG (SAM-40); adapted 16+6 channels for Neuro-GPT
- **Methodology / Model:** Preprocessing + Neuro-GPT Fine-Tuning + ML Comparison
- **Reported performance:** 74.4%
- **Identified gap / limitation:** Dataset lacked clear stress–relax distinction; self-reported stress ratings may be subjective; electrode mismatch between Neuro-GPT and SAM-40.
- **Citation as provided in the source table:** Lloyd, C. et al. (2024) ‘Stress-GPT: Stress detection with an EEG-based foundation model’, Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, pp. 2341–2346. doi:10.1145/3636534.3698121.

### 4. S. Raj, N. N. Sakhare, Y. K. Sharma, M. Soni, A. Bhange, P. K. Deshmukh (2023)
- **Dataset:** EEG Brainwave Dataset: Feeling Emotions
- **Modality:** Multi-channel EEG
- **Methodology / Model:** Preprocessing + CNN/LSTM/ConvLSTM + PSO
- **Reported performance:** 99.6%
- **Identified gap / limitation:** Need for improved precision and efficiency; future scope includes multimodal integration for stronger emotion recognition.
- **Citation as provided in the source table:** Raj, S. et al. (2023) ‘Deep learningbased emotion detection from EEG signals for humanmachineinteraction’, Proceedings of the 5th International Conference on Information Management & Machine Intelligence, pp. 1–6. doi:10.1145/3647444.3647930.

### 5. A. Kumar, A. Mathur, D. Joshi (2025)
- **Dataset:** 18 participants
- **Modality:** EEG (0.1–45 Hz, 200 Hz sampling)
- **Methodology / Model:** FFT + PCA + SNN + XGBoost + SVC
- **Reported performance:** 60% (2-class), 43% (3-class)
- **Identified gap / limitation:** Feature extraction approach may limit performance; need better real-time EEG representation to fully utilise SNN potential.
- **Citation as provided in the source table:** Kumar, A., Mathur, A. and Joshi, D. (2025) ‘Connecting brains and interfaces: Real-time EEG-based stress detection via spiking neural networks’, Proceedings of the ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies, pp. 666–671. doi:10.1145/3715335.3736311.

### 6. F. Zhu, B. Wu, Y. Huo, R. Dang, B. Hu, Q. Wang (2025)
- **Dataset:** 42 adolescents
- **Modality:** 64-channel EEG + Eye Tracking
- **Methodology / Model:** EVT + RGAT + MSNet Fusion
- **Reported performance:** 93.62%
- **Identified gap / limitation:** High multimodal acquisition cost; limited dataset size; need simpler acquisition systems.
- **Citation as provided in the source table:** Zhu, F. et al. (2025) ‘MSNet: Multimodal self-attention network for depression detection via fusion of eye tracking and EEG’, Proceedings of the 2025 Symposium on Eye Tracking Research and Applications, pp. 1–6. doi:10.1145/3715669.3723106.

### 7. Z. Yuan, Y. Yang, H. Liu (2025)
- **Dataset:** DEAP Dataset (32 subjects)
- **Modality:** 12-channel EEG
- **Methodology / Model:** Nonlinear EEG Features + C4.5 Decision Tree
- **Reported performance:** 86.39%
- **Identified gap / limitation:** Cross-modal fusion and complex emotion recognition remain challenging; need efficient real-time feedback and stronger privacy/safety mechanisms.
- **Citation as provided in the source table:** Yuan, Z., Yang, Y. and Liu, H. (2025) ‘Robotic recognition of human depression based on nonlinear EEG features’, Proceedings of the 2025 International Conference on Generative Artificial Intelligence for Business, pp. 430–435. doi:10.1145/3766918.3766988.

### 8. Y. Xue, M. Zordan, X. Zhang, C. Chen (2024)
- **Dataset:** 20 participants
- **Modality:** Single-channel EEG (NeuroSky TGAM)
- **Methodology / Model:** EEG Biofeedback + Stress Visualization
- **Reported performance:** Not reported in the paper
- **Identified gap / limitation:** Future work includes integrating additional biosignals and quantitative stress assessment.
- **Citation as provided in the source table:** Xue, Y. et al. (2024) ‘Minus X Minus = Plus: A semi-controlled data processing system based on EEG biofeedback for stress sharing’, Proceedings of the Twelfth International Symposium of Chinese CHI, pp. 774–780. doi:10.1145/3758871.3758963.

### 9. A. Rezgui (2025)
- **Dataset:** CogWear Dataset (11 volunteers)
- **Modality:** 4-channel EEG + BVP + EDA + Temperature
- **Methodology / Model:** FFT + Granger Causality + Stroop Test
- **Reported performance:** Not reported in the paper
- **Identified gap / limitation:** Causation with EDA and temperature was inconclusive; future work aims to identify stronger causation with other vital signs.
- **Citation as provided in the source table:** Rezgui, A. (2025) ‘On the causality between cognitive stress and physiological stress: The Stroop test as a case study’, Adjunct Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization, pp. 495–502. doi:10.1145/3708319.3734838.

### 10. Q. Liu, W. Ye, L. Zhang, Z. Liang (2025)
- **Dataset:** SEED, SEED-IV, DEAP
- **Modality:** Multi-channel EEG
- **Methodology / Model:** SCMM + Soft Contrastive Learning + Hybrid Masking
- **Reported performance:** 91.70%
- **Identified gap / limitation:** Limited cross-corpus generalization in existing methods; reliance on traditional contrastive learning and domain adaptation reduces generalizability.
- **Citation as provided in the source table:** Liu, Q. et al. (2025) ‘EEG-SCMM: Soft contrastive masked modeling for cross-corpus EEG-based emotion recognition’, Proceedings of the 33rd ACM International Conference on Multimedia, pp. 5834–5842. doi:10.1145/3746027.3755798.

### 11. Y. Wang, F. Liu, L. Yang (2021)
- **Dataset:** MODMA Dataset (53 subjects)
- **Modality:** 128-channel EEG
- **Methodology / Model:** ITD + TCN (L-Softmax)
- **Reported performance:** 86.87%
- **Identified gap / limitation:** CNN and RNN have limitations in temporal modeling and parallel computation; improved discriminative feature learning is needed.
- **Citation as provided in the source table:** Wang, Y., Liu, F. and Yang, L. (2021) ‘EEG-based depression recognition using intrinsic time-scale decomposition and temporal convolution network’, The Fifth International Conference on Biological Information and Biomedical Engineering, pp. 1–6. doi:10.1145/3469678.3469683.

### 12. W. Zhang, W. Liu, X. Liu, S. Liu, D. Ming (2022)
- **Dataset:** 63 subjects (30 MDD + 33 HC)
- **Modality:** 64-channel EEG (40Hz-ASSR)
- **Methodology / Model:** ITC + TGC + SVM
- **Reported performance:** 81.8%
- **Identified gap / limitation:** Objective biomarkers for MDD are still lacking; ASSR abnormalities in MDD remain underexplored; TGC in MDD has not been fully explored.
- **Citation as provided in the source table:** Zhang, W. et al. (2022) ‘Altered lateralization of gamma oscillations and theta-gamma coupling in major depression: An EEG study’, Proceedings of the 2022 9th International Conference on Biomedical and Bioinformatics Engineering, pp. 203–208. doi:10.1145/3574198.3574230.

### 13. N. Sakib, T. Faruk, M. K. Islam (2023)
- **Dataset:** 40 participants
- **Modality:** 14-channel wireless EEG
- **Methodology / Model:** Frequency Filtering + Nonlinear Features + SVM
- **Reported performance:** 94.88%
- **Identified gap / limitation:** GAD-7 based EEG screening and young-adult anxiety studies are limited.
- **Citation as provided in the source table:** Sakib, N., Faruk, T. and Islam, M.K. (2023) ‘Wireless EEG based anxiety screening among young adults using machine learning model’, Proceedings of the 2023 8th International Conference on Biomedical Imaging, Signal Processing, pp. 97–103. doi:10.1145/3634875.3634889.

### 14. C. Liang, Y. Ju, X. Hao, Y. Zhang, S. Liu (2023)
- **Dataset:** 20 subjects (10 GAD + 10 HC)
- **Modality:** 64-channel resting-state EEG
- **Methodology / Model:** Power Spectrum + KNN/LDA/SVM
- **Reported performance:** 69.76%
- **Identified gap / limitation:** Anxiety subtypes not differentiated; depression comorbidity not considered; small sample size; only abnormal channels analyzed; future work suggested broader features and brain-wide analysis.
- **Citation as provided in the source table:** Liang, C. et al. (2023) ‘Abnormal oscillations in beta and gamma of generalized anxiety disorder patients based on resting state EEG’, Proceedings of the 2023 10th International Conference on Biomedical and Bioinformatics Engineering, pp. 201–207. doi:10.1145/3637732.3637741.

### 15. H. Liu, Y. Hu (2025)
- **Dataset:** SEED Dataset (8 subjects tested)
- **Modality:** Multi-channel EEG
- **Methodology / Model:** PSD + Improved EEG-TCNet + SVM
- **Reported performance:** 81.08%
- **Identified gap / limitation:** Existing EEG emotion recognition methods still require improved signal recognition accuracy.
- **Citation as provided in the source table:** Liu, H. and Hu, Y. (2025) ‘EEG emotion signal recognition method based on improved EEG-tcnet’, Proceedings of the 2025 3rd International Conference on Internet of Things and Cloud Computing Technology, pp. 302–307. doi:10.1145/3776865.3776914.

### 16. Shikha, M. Agrawal, M. A. Anwar, D. Sethia (2021)
- **Dataset:** DASPS Dataset (23 participants)
- **Modality:** 14-channel wireless EEG
- **Methodology / Model:** Feature Extraction + RFECV + SSAE + Decision Tree
- **Reported performance:** 83.93%
- **Identified gap / limitation:** Existing anxiety classification methods require improved accuracy and reduced feature redundancy.
- **Citation as provided in the source table:** Shikha et al. (2021) ‘Stacked sparse Autoencoder and machine learning based anxiety classification using EEG signals’, The First International Conference on AI-ML-Systems, pp. 1–7. doi:10.1145/3486001.3486227.

### 17. S. Zhang, Y. Hu, X. Yi, S. Nanayakkara, X. Chen (2025)
- **Dataset:** Multimodal physiological data
- **Modality:** EEG + Text + Images
- **Methodology / Model:** IntervEEG-LLM + Multimodal Modeling + Mental Health Intervention
- **Reported performance:** Not reported in the paper
- **Identified gap / limitation:** EEG-based LLM intervention remains underexplored; real-world deployment, privacy, and additional physiological integration require further study.
- **Citation as provided in the source table:** Zhang, S. et al. (2025) ‘Interveeg-LLM: Exploring EEG-based multimodal data for customized mental health interventions’, Companion Proceedings of the ACM on Web Conference 2025, pp. 2320–2326. doi:10.1145/3701716.3717550.

---

# Source 2 — Comparison_Table_2.pdf

**Source-table categories:** EEG Stress | EEG Depression | Multimodal | Survey / Gap

### 1. Hatem S. A. Hamatta et al. (2022)
- **Category:** EEG Stress
- **Dataset:** DEAP Dataset
- **Modality:** EEG + PPG + GSR
- **Methodology / Model:** Genetic Algorithm + ECNN-LSTM
- **Reported performance:** 99.9%
- **Identified gap / limitation:** Limited comparison with advanced feature extraction approaches; future enhancement required for improving classification robustness and feature optimization.

### 2. Deep Learning Mental Stress Prediction paper authors (2025)
- **Category:** EEG Stress
- **Dataset:** EEG Psychiatric Disorders Dataset (Kaggle)
- **Modality:** EEG
- **Methodology / Model:** AFAWT + STFT-RT + YDSE + ParNeXt v1
- **Reported performance:** Optimized (exact % not reported)
- **Identified gap / limitation:** EEG noise sensitivity, subject variability, environmental interference, classifier overfitting, and large labeled-data requirements reduce robustness and real-time applicability.

### 3. Md Raihan Khan, Mohiuddin Ahmad (2024)
- **Category:** EEG Stress
- **Dataset:** GAMEEMO + SEED
- **Modality:** EEG
- **Methodology / Model:** Random Forest + RNN + PSD Features
- **Reported performance:** RNN: 87% (Arousal), 83% (Valence)
- **Identified gap / limitation:** EEG non-stationarity, noise, feature extraction complexity, and limited research specifically targeting academic mental stress remain challenges.

### 4. R. Gupta, M. Alam, P. Agarwal (2020)
- **Category:** EEG Stress
- **Dataset:** EEG Stress Dataset
- **Modality:** EEG
- **Methodology / Model:** Modified SVM + Whale Optimization
- **Reported performance:** Not explicitly reported
- **Identified gap / limitation:** Scalability limitations and reduced generalization capability across diverse EEG stress scenarios.

### 5. Comparative ML Mental Stress Detection Paper (2024)
- **Category:** EEG Stress
- **Dataset:** EEG Dataset
- **Modality:** EEG
- **Methodology / Model:** KNN + SVM + RF + EDA analysis
- **Reported performance:** KNN: lowest MAE (0.62)
- **Identified gap / limitation:** Small and noisy datasets limit deep learning adoption; model robustness can improve through stronger preprocessing and larger datasets.

### 6. Nazmus Sakib, Md Kaful Islam, Tasnuva Faruk (2023)
- **Category:** EEG Depression
- **Dataset:** Custom EEG Dataset (32 subjects), PHQ-9
- **Modality:** EEG
- **Methodology / Model:** KNN + SVM + Hjorth Parameters + Shannon Entropy + Log Energy Entropy
- **Reported performance:** 98.43%
- **Identified gap / limitation:** Small participant pool (32 subjects) and limited demographic diversity constrain broader clinical generalization.

### 7. AI Analysis EEG Depression paper authors (2024)
- **Category:** EEG Depression
- **Dataset:** Clinical EEG + Evoked Potential observations
- **Modality:** EEG + Evoked Potential
- **Methodology / Model:** AI-assisted EEG analysis
- **Reported performance:** Diagnostic support (no % reported)
- **Identified gap / limitation:** EEG biomarkers alone may not provide sufficient diagnostic reliability; broader multimodal clinical evaluation is needed.

### 8. EEG ML/DL Depression Survey authors (2025)
- **Category:** EEG Depression
- **Dataset:** Multiple EEG Depression papers
- **Modality:** EEG
- **Methodology / Model:** CNN + RNN + Hybrid DL + Survey Analysis
- **Reported performance:** N/A (Survey Paper)
- **Identified gap / limitation:** Heavy reliance on binary classification and insufficient severity-level prediction; multimodal integration remains underexplored.

### 9. Weitong Guo et al. (2026)
- **Category:** Multimodal
- **Dataset:** AVEC2014 + Chinese localized depression dataset
- **Modality:** Audio + Video
- **Methodology / Model:** Transformer Cross Modal Coattention Network (CMCAN)
- **Reported performance:** 83.9%
- **Identified gap / limitation:** Cross-modal temporal synchronization and preservation of modality-specific information remain difficult in multimodal fusion.

### 10. IIOT Stress Assessment paper authors (2025)
- **Category:** Multimodal
- **Dataset:** Proprietary multimodal sensing framework
- **Modality:** EEG + HRV + GSR + Voice + Behavioral + Environment
- **Methodology / Model:** Transformer Attention + Federated Learning + Multimodal Fusion
- **Reported performance:** +3.6% improvement over TFN
- **Identified gap / limitation:** Privacy constraints, multimodal synchronization complexity, noisy sensor conditions, and real-world deployment challenges affect robustness.

### 11. Hakan Uyanik et al. (2025)
- **Category:** Survey / Gap
- **Dataset:** 74 reviewed studies
- **Modality:** EEG
- **Methodology / Model:** ML + DL Systematic Review
- **Reported performance:** N/A (Review Paper)
- **Identified gap / limitation:** Larger EEG datasets, improved model interpretability, and stronger generalization across populations are needed.

### 12. EEG-MDD Survey paper authors (2025)
- **Category:** Survey / Gap
- **Dataset:** 31 EEG-MDD papers
- **Modality:** EEG
- **Methodology / Model:** CNN + RNN + LSTM + Hybrid DL Survey
- **Reported performance:** N/A (Survey Paper)
- **Identified gap / limitation:** Small homogeneous datasets and limited demographic diversity reduce model generalization capability.

### 13. EEG ML/DL Depression Survey authors (2025)
- **Category:** Survey / Gap
- **Dataset:** Multiple EEG studies
- **Modality:** EEG
- **Methodology / Model:** Survey + Gap Analysis
- **Reported performance:** N/A (Survey Paper)
- **Identified gap / limitation:** Limited multimodal fusion strategies and inadequate depression severity-stage prediction remain open research directions.

---

# Source 3 — Comparison_Table_3.pdf

> The source is titled **“IEEE Research Papers (2021–2026)”**, but it also contains a 2012 entry and multiple entries whose year/authors/dataset are marked as not specified. Those details are preserved as given.

### 1. S. A. Bakry et al. (2023)
- **Dataset:** Multiple (Review)
- **Modality:** Video, EEG
- **Methodology / Model:** Machine Learning pipelines (CAD systems)
- **Reported performance:** N/A
- **Identified gap / limitation:** Lack of large-scale multimodal datasets; need for real-time clinical validation in everyday settings.

### 2. S. T. Noor et al. (2023)
- **Dataset:** PhysioNet / Pervasive EEG
- **Modality:** EEG, ECG
- **Methodology / Model:** LSTM Autoencoder with RNN
- **Reported performance:** 97
- **Identified gap / limitation:** High computational cost of LSTM frameworks; limited to specific demographics.

### 3. Not specified (2022)
- **Dataset:** Not specified
- **Modality:** EEG
- **Methodology / Model:** Sentiment Analysis via Brain Signals
- **Reported performance:** Not specified
- **Identified gap / limitation:** Vulnerability to noise and motion artifacts in EEG signals; small subject pools.

### 4. S. Koelstra et al. (2012)
- **Dataset:** DEAP Database
- **Modality:** EEG, Physiological, Video
- **Methodology / Model:** Single-trial classification & Affective state estimation
- **Reported performance:** Baseline
- **Identified gap / limitation:** Subjective rating bias (self-reporting); laboratory stimuli don't perfectly reflect natural environments.

### 5. M. M. R. Khan et al. (2024)
- **Dataset:** Math Task (23 Subjects)
- **Modality:** EEG
- **Methodology / Model:** SVM and Ensemble Learning with optimized pre-processing
- **Reported performance:** 96.85
- **Identified gap / limitation:** Handcrafted features limit automated scalability; tested exclusively on mathematical stressors.

### 6. S. K. Yadav et al. (2023)
- **Dataset:** Audio/Speech Samples
- **Modality:** Audio, EEG
- **Methodology / Model:** Filter Feature Selection (RELIEFF), KNN, Naive Bayes
- **Reported performance:** 88
- **Identified gap / limitation:** Data synchronization issues between audio and EEG modalities; relatively lower accuracy.

### 7. Md. M. R. Khan et al. (2025)
- **Dataset:** SAM-40 (Math Task)
- **Modality:** EEG
- **Methodology / Model:** DWT, Filter bank, Two-stage hybrid classifier
- **Reported performance:** 95.56
- **Identified gap / limitation:** Small sample size required data augmentation; generalizability to other stressors is unproven.

### 8. A. Singh et al. (2026)
- **Dataset:** MODMA
- **Modality:** EEG, Audio
- **Methodology / Model:** ResKAN encoder, Cross-attention mechanism, KAN
- **Reported performance:** 99
- **Identified gap / limitation:** High computational resource demands for KAN networks; pending diverse clinical trials.

### 9. Not specified (2023)
- **Dataset:** Frontal Lobe (38 Subjects)
- **Modality:** EEG
- **Methodology / Model:** Majority Voting Ensemble (SVM, KNN, Naïve Bayes)
- **Reported performance:** 93.85
- **Identified gap / limitation:** Heavy reliance on manual feature extraction limits fully automated scaling.

### 10. Not specified (2023)
- **Dataset:** VR Horror / IQ Test
- **Modality:** EEG
- **Methodology / Model:** K-Means Clustering + Support Vector Machine (SVM)
- **Reported performance:** Not specified
- **Identified gap / limitation:** Lack of objective ground truth labels; heavily relies on subjective evaluations.

### 11. Nazmus Sakib et al. (2026)
- **Dataset:** IUB Students (PSS-10)
- **Modality:** EEG
- **Methodology / Model:** Bagged Trees Classifier
- **Reported performance:** Not specified
- **Identified gap / limitation:** Generalizability is limited as participants were exclusively university students.

### 12. Vaishali Darbar et al. (2024)
- **Dataset:** Not specified
- **Modality:** EEG
- **Methodology / Model:** Deep Neural Network (DNN) & ML Classification
- **Reported performance:** Not specified
- **Identified gap / limitation:** The "black-box" nature of deep neural networks lacks interpretability for clinical diagnosis.

### 13. Not specified (Not specified)
- **Dataset:** Not specified
- **Modality:** EEG
- **Methodology / Model:** Power Spectral Density (PSD) Asymmetry Analysis
- **Reported performance:** N/A
- **Identified gap / limitation:** High individual variability makes establishing standard universal baselines difficult.

### 14. Not specified (Not specified)
- **Dataset:** Multiple (Review)
- **Modality:** Nano EEG
- **Methodology / Model:** Review of Management and Detection Techniques
- **Reported performance:** N/A
- **Identified gap / limitation:** Manufacturing complexities; long-term biocompatibility and durability of nano-sensors are largely unknown.

### 15. Not specified (Not specified)
- **Dataset:** Not specified
- **Modality:** EEG
- **Methodology / Model:** Interpretable Machine Learning Framework
- **Reported performance:** Not specified
- **Identified gap / limitation:** Balancing high predictive accuracy with interpretability across diverse demographics remains a challenge.

### 16. Anagha Acharya et al. (Not specified)
- **Dataset:** Multiple (Review)
- **Modality:** EEG
- **Methodology / Model:** Review of ML clinical tools for Postpartum Depression
- **Reported performance:** N/A
- **Identified gap / limitation:** Scarcity of longitudinal EEG data during and post-pregnancy due to ethical data collection constraints.

### 17. Urvashi P. Shukla et al. (Not specified)
- **Dataset:** Not specified
- **Modality:** EEG
- **Methodology / Model:** Deep Neural Network (DeepNet)
- **Reported performance:** Not specified
- **Identified gap / limitation:** Multi-channel setups are cumbersome and intrusive for daily wear and real-world monitoring.

### 18. Yashada Nikam et al. (Not specified)
- **Dataset:** Multiple (Survey)
- **Modality:** Physiological/EEG
- **Methodology / Model:** Survey of Feature Extraction & ML models
- **Reported performance:** N/A
- **Identified gap / limitation:** Lack of standardized pre-processing pipelines makes cross-study comparisons difficult.

### 19. Not specified (Not specified)
- **Dataset:** Multiple (Review)
- **Modality:** EEG
- **Methodology / Model:** Review of Current Strategies and Future Aspects
- **Reported performance:** N/A
- **Identified gap / limitation:** Transitioning from lab-grade equipment to reliable consumer-grade wearables with high signal-to-noise ratios.

### 20. Natalya Milakhina et al. (2020)
- **Dataset:** Siberian Migrants
- **Modality:** Resting State EEG
- **Methodology / Model:** Delta/Gamma Spectral Marker Analysis
- **Reported performance:** N/A
- **Identified gap / limitation:** Confounding variables (cultural/lifestyle changes) not fully isolated; highly specific to polar regions.

### 21. S. R. Chowdhury et al. (2023)
- **Dataset:** Rat Model
- **Modality:** EEG, tDCS
- **Methodology / Model:** Preclinical Electrode Validation
- **Reported performance:** N/A
- **Identified gap / limitation:** Direct translation from animal (rat) models to human clinical trials is complex; long-term effects of tDCS need tracking.

---

# Cross-source research-gap synthesis

The following themes are a **synthesis of the gaps explicitly listed in the three uploaded tables**. They are not additional claims from external sources.

1. **Small, homogeneous, or narrowly sampled datasets.** Several entries flag small participant pools, limited demographics, university-only samples, or restricted populations, reducing confidence in cross-population generalization.
2. **Generalization across datasets, subjects, and stressors.** Cross-corpus performance, transfer to different stressors, and robustness across diverse EEG scenarios repeatedly appear as unresolved issues.
3. **EEG noise, non-stationarity, and preprocessing dependence.** Multiple studies identify motion artifacts, signal noise, feature-extraction complexity, and the lack of standardized preprocessing as barriers to robust deployment.
4. **Reliance on handcrafted or manually extracted features.** Several classical ML pipelines depend on manual feature engineering, which can limit scalability and may make performance sensitive to feature quality.
5. **Multimodal fusion and synchronization.** Combining EEG with audio, video, eye tracking, HRV, GSR, EDA, temperature, text, images, or environmental signals introduces synchronization, modality-preservation, cost, and robustness challenges.
6. **Computational cost and real-time feasibility.** LSTM-based systems, KAN-based systems, and other complex architectures may impose resource demands that complicate real-time or wearable use.
7. **Interpretability and clinically meaningful biomarkers.** Black-box behavior, limited interpretability, and the need for objective or reliable biomarkers recur across depression and broader mental-health applications.
8. **Subjective labels and weak ground truth.** Self-reported stress/emotion ratings and subjective evaluation can introduce label uncertainty; some studies explicitly call for more objective quantitative assessment.
9. **Wearability and acquisition burden.** High-channel-count EEG, multimodal hardware, and lab-grade sensing can be cumbersome or expensive; some entries call for simpler or consumer-grade acquisition systems.
10. **Privacy, safety, and real-world deployment.** Privacy, safety, federated-learning constraints, deployment robustness, and validation in everyday or clinical settings are repeatedly identified as open challenges.
11. **Limited severity-level or fine-grained prediction.** Survey entries note heavy reliance on binary classification and insufficient prediction of depression severity or more complex affective states.
12. **Need for broader validation.** Multiple sources call for larger datasets, diverse clinical trials, longitudinal data, and stronger real-world validation before broad clinical use.

## Potential literature-review organization

- **Conventional / feature-engineered EEG approaches:** PSD, FFT, entropy, Hjorth parameters, nonlinear features, SVM, KNN, Decision Trees, ensemble methods.
- **Deep-learning approaches:** CNN, 1D-CNN, LSTM/RNN, ConvLSTM, TCN, autoencoders, SNN, EEG-TCNet.
- **Foundation / advanced representation approaches:** Neuro-GPT, contrastive masked modeling, transformer attention, cross-attention, KAN/ResKAN, IntervEEG-LLM.
- **Multimodal approaches:** EEG fused with eye tracking, PPG, GSR/EDA, HRV, audio, video, text, images, temperature, and environmental/behavioral signals.
- **Deployment and translational gaps:** generalization, interpretability, privacy, wearable acquisition, real-time inference, clinical validation, and longitudinal evidence.

## Citation-quality cautions

- Source 1 provides formal citation strings for all 17 rows; these should still be checked against the original publication metadata before final submission.
- Sources 2 and 3 contain several generic labels, incomplete author fields, missing years, or missing formal citations. Treat those rows as **leads for literature review**, not submission-ready bibliography entries.
- Reported accuracy/performance values should be described with the corresponding task and dataset rather than compared as a single leaderboard.
- The source tables mix stress, anxiety, depression, emotion recognition, stroke classification, surveys, preclinical work, and multimodal systems. Select only studies that directly support the scope of the paper section you are writing.
