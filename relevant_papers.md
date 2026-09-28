# Literature support: papers, DOIs and relevant sections

Every row is a claim or technique the report needs to support, the paper that supports it, its
DOI, and the part of the paper that carries it. The "Used in" column points at the report
section in `fyp_report.md`.

All DOIs marked verified were resolved through the Crossref API on 2026-09-28 and the returned
title was matched against the intended paper. Section references follow the published version;
check the number against the copy you actually read, and where the section is given by content
rather than number it means the numbering was not confirmed.

---

## 1. Screening, CAD and clinical context

| Claim or technique                                                                        | Paper                                                          | DOI                               | Relevant section                                                                               | Used in       |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- | ------------- |
| CAD in screening raises recall and the false-positive rate without a clear detection gain | Gilbert et al. 2008, *NEJM*                                    | 10.1056/NEJMoa0803545             | the trial results and the discussion of reader recall against false positives                  | 2.1           |
| AI can read screening mammograms at reader level, with reader-workload implications       | McKinney et al. 2020, *Nature*                                 | 10.1038/s41586-019-1799-6         | the UK and US reader-study results and Methods, study design                                   | 1.1, 2.1, 6.3 |
| Randomised evidence on AI-supported screen reading and interval cancers                   | Lang et al. 2023, MASAI, *Lancet Oncology*                     | 10.1016/S1470-2045(23)00298-X     | the primary outcome (interval cancers) and the reader-replacement arm with its safety analyses | 1.1, 2.1, 6.3 |
| Prospective paired-reader non-inferiority and the false-positive cost                     | Dembrower et al. 2023, ScreenTrustCAD, *Lancet Digital Health* | 10.1016/S2589-7500(23)00153-X     | the paired-reader design, the non-inferiority analysis and the safety analysis                 | 2.1, 6.3      |
| High sensitivity at a low false-positive rate is achievable                               | Lotter et al. 2021, *Nature Medicine*                          | 10.1038/s41591-020-01174-9        | the screening results reported at fixed operating points                                       | 2.1, 5.7      |
| Commercial AI generalises imperfectly to another screening population                     | Salim et al. 2020, *JAMA Oncology*                             | 10.1001/jamaoncol.2020.3321       | the external evaluation on an independent screening population                                 | 2.1, 2.8, 5.7 |
| Performance is site- and vendor-dependent, so robustness has to be measured               | Yala et al. 2022, *Science Translational Medicine*             | 10.1126/scitranslmed.aba4373      | the multi-institution evaluation and the robustness and calibration analysis                   | 2.8, 5.8      |
| CAD has a long false-positive problem and what that means for a new system                | Nishikawa 2007, *Computerized Medical Imaging and Graphics*    | 10.1016/j.compmedimag.2007.02.009 | the review of CAD performance and the false-positive limitations                               | 2.1           |

## 2. Datasets and preprocessing

| Claim or technique                                                                 | Paper                                                     | DOI                          | Relevant section                                                            | Used in       |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------- | ------------- |
| CBIS-DDSM curation, ROI masks and pathology labels, and the dataset's known issues | Lee et al. 2017, *Scientific Data*                        | 10.1038/sdata.2017.177       | Methods (curation, ROI generation, pathology labels) and the Usage Notes    | 2.2, 3.4, 4.2 |
| Ground-truth boxes are derived from lesion masks, not from reader-drawn boxes      | Lee et al. 2017                                           | 10.1038/sdata.2017.177       | the ROI mask description in Methods                                         | 3.4, 5.8      |
| Benign versus malignant labels, including BENIGN_WITHOUT_CALLBACK                  | Lee et al. 2017                                           | 10.1038/sdata.2017.177       | the pathology label definitions in Methods and the dataset documentation    | 2.7, 3.4      |
| Breast region masking and boundary handling                                        | Rampun et al. 2017, *Artificial Intelligence in Medicine* | 10.1016/j.artmed.2017.06.001 | the fully automated breast boundary and pectoral muscle segmentation method | 2.2, 4.2      |
| Threshold selection for masking (the technique family used)                        | Otsu 1979, *IEEE Trans. SMC*                              | 10.1109/TSMC.1979.4310076    | the threshold selection criterion                                           | 4.2           |

## 3. Deep learning for mammography and benchmark numbers

| Claim or technique                                                                              | Paper                                                                      | DOI                              | Relevant section                                                                                            | Used in       |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------- |
| Survey of deep learning for mammography, with the reported detection numbers to compare against | Wang 2024, *Frontiers in Oncology*                                         | 10.3389/fonc.2024.1281922        | the architecture review and the comparison table of reported detection results                              | 2.3, 5.7      |
| Ensembles and stacking in mammography classification, and what is usually reported              | Fajrin and Min 2025, *Diagnostics*                                         | 10.3390/diagnostics15222829      | the ensemble taxonomy and the pooled metric comparison                                                      | 2.6, 5.7      |
| Shallow against deep CNNs for breast cancer detection                                           | Das et al. 2023, *Frontiers in Genetics*                                   | 10.3389/fgene.2022.1097207       | the architecture comparison and its results                                                                 | 2.3           |
| Explainability of YOLO detectors on mammograms                                                  | Ariyametkul and Paing 2025, *Quantitative Imaging in Medicine and Surgery* | 10.21037/qims-2024-2911          | the heat-map explainability analysis and its clinical reading                                               | 2.3, 4.9, 5.6 |
| Sliding-window classification plus a cascade is the reference mammography pipeline              | Ribli et al. 2018, *Scientific Reports*                                    | 10.1038/s41598-018-22437-z       | the object detection method and the classification cascade                                                  | 2.3, 2.4, 5.7 |
| CNN CAD with candidate extraction, evaluated at fixed false positives per image                 | Kooi et al. 2017, *Medical Image Analysis*                                 | 10.1016/j.media.2016.07.007      | the candidate extraction and CNN classification, and the performance reported at fixed false-positive rates | 2.3, 5.7      |
| Localisation followed by classification as separate stages                                      | Shen et al. 2019, *Scientific Reports*                                     | 10.1038/s41598-019-48995-4       | the two-stage architecture and the operating-point results                                                  | 2.3, 5.7      |
| Integrated detection, segmentation and classification pipeline                                  | Al-antari et al. 2018, *Int. J. Medical Informatics*                       | 10.1016/j.ijmedinf.2018.06.003   | the region-based DCNN detection and classification pipeline                                                 | 2.3, 5.7      |
| YOLO used for mass detection and classification in mammograms                                   | Al-Masni et al. 2018, *Computer Methods and Programs in Biomedicine*       | 10.1016/j.cmpb.2018.01.017       | the YOLO detection stage and the classification stage with sensitivity and false positives per image        | 2.3, 2.5, 5.7 |
| Mass detection reported with FROC-style and AP evaluation                                       | Agarwal et al. 2020, *Computers in Biology and Medicine*                   | 10.1016/j.compbiomed.2020.103774 | the mass detection network and the FROC or AP evaluation                                                    | 2.3, 5.7      |
| Multi-stage deep learning for mass analysis and its evaluation protocol                         | Dhungel et al. 2017, *Medical Image Analysis*                              | 10.1016/j.media.2017.01.009      | the multi-stage pipeline and the evaluation protocol                                                        | 2.3, 5.7      |

## 4. Detector architectures and training objectives

| Claim or technique                                                           | Paper                                           | DOI                         | Relevant section                                                                                                     | Used in            |
| ---------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Region proposals plus a classifier, and greedy NMS at test time              | Girshick et al. 2014, R-CNN, *CVPR*             | 10.1109/CVPR.2014.81        | the region proposal and CNN feature pipeline, and test-time detection with NMS                                       | 2.4, 2.6, 4.7      |
| RoI pooling, multi-task loss with a robust box loss, and mini-batch sampling | Girshick 2015, Fast R-CNN, *ICCV*               | 10.1109/ICCV.2015.169       | RoI pooling, the multi-task loss, and mini-batch sampling                                                            | 2.4, 4.3           |
| Anchors, positive and negative IoU bands, and the RPN classification loss    | Ren et al. 2015, Faster R-CNN, *IEEE TPAMI*     | 10.1109/TPAMI.2016.2577031  | anchors and IoU-based labels, the loss function, and RPN training                                                    | 2.4, 2.5, 4.3, 4.4 |
| Multi-scale feature fusion with lateral connections                          | Lin et al. 2017, FPN, *CVPR*                    | 10.1109/CVPR.2017.106       | bottom-up path, top-down path and lateral connections                                                                | 2.4, 4.3, 4.4      |
| Dense anchors with a focal loss for extreme foreground-background imbalance  | Lin et al. 2017, RetinaNet, *ICCV*              | 10.1109/ICCV.2017.324       | the focal loss derivation and the anchor settings                                                                    | 2.4, 2.5, 4.3      |
| Multi-scale anchor design, matching strategy and hard negative mining        | Liu et al. 2016, SSD, *ECCV*                    | 10.1007/978-3-319-46448-0_2 | the training section (matching strategy, hard negative mining, augmentation) and the scale and aspect ratio settings | 2.4, 2.5, 4.3      |
| Single-network grid prediction with NMS at inference                         | Redmon et al. 2016, YOLO, *CVPR*                | 10.1109/CVPR.2016.91        | the unified detection design, training and inference                                                                 | 2.4, 4.5           |
| Mosaic augmentation as used by the YOLO recipe                               | Bochkovskiy et al. 2020, YOLOv4, arXiv preprint | see section 9               | the bag-of-freebies section describing mosaic augmentation                                                           | 2.5, 4.5           |
| YOLO26 architectural changes used without modification                       | Ultralytics YOLO26 paper cited in the brief     | see section 9               | the architectural enhancements and benchmark section                                                                 | 2.4, 4.5           |
| MobileNetV3 blocks as the detection backbone                                 | Howard et al. 2019, *ICCV*                      | 10.1109/ICCV.2019.00140     | the block design and the detection backbone configuration                                                            | 4.4                |
| Residual connections for trainable depth                                     | He et al. 2016, ResNet, *CVPR*                  | 10.1109/CVPR.2016.90        | residual learning and identity mappings                                                                              | 2.4                |

## 5. Training, regularisation and loss design

| Claim or technique                                                                   | Paper                                           | DOI                          | Relevant section                                                           | Used in       |
| ------------------------------------------------------------------------------------ | ----------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------- | ------------- |
| GroupNorm is stable at small batch sizes where batch statistics are unreliable       | Wu and He 2018, *ECCV*                          | 10.1007/978-3-030-01261-8_1  | the GroupNorm formulation and the small-batch comparison against BatchNorm | 2.5, 4.3      |
| Batch normalisation uses mini-batch statistics, which is what fails at batch 4       | Ioffe and Szegedy 2015, arXiv preprint          | see section 9                | normalisation via mini-batch statistics                                    | 2.5, 4.3      |
| Decoupled weight decay and why it is applied to kernel weights only                  | Loshchilov and Hutter 2019, AdamW, ICLR         | see section 9                | decoupled weight decay and its algorithm                                   | 2.5, 4.3      |
| Adaptive moment estimation, the base optimiser                                       | Kingma and Ba 2015, Adam, ICLR                  | see section 9                | the algorithm and its update rules                                         | 4.3           |
| Cosine annealing of the learning rate                                                | Loshchilov and Hutter 2017, SGDR, ICLR          | see section 9                | the cosine schedule and its restarts                                       | 2.5, 4.3      |
| Gradual warmup at the start of training                                              | Goyal et al. 2017, arXiv preprint               | see section 9                | the gradual warmup section                                                 | 2.5, 4.3      |
| Label smoothing as a regulariser for the class term                                  | Szegedy et al. 2016, *CVPR*                     | 10.1109/CVPR.2016.308        | model regularization via label smoothing                                   | 2.5, 4.3      |
| Robust location estimation as the basis of a Huber-style box loss                    | Huber 1964, *Annals of Mathematical Statistics* | 10.1214/aoms/1177703732      | the robust estimator definition                                            | 2.5, 4.3      |
| Smooth L1, the detection-specific form of the box loss                               | Girshick 2015, Fast R-CNN                       | 10.1109/ICCV.2015.169        | the multi-task loss section                                                | 4.3           |
| Hard examples can be mined online inside a detector's training loop                  | Shrivastava et al. 2016, OHEM, *CVPR*           | 10.1109/CVPR.2016.89         | the OHEM algorithm and its application to detection                        | 2.5, 4.3, 6.4 |
| Class imbalance handling by sampling and thresholding, and what is actually measured | Buda et al. 2018, *Neural Networks*             | 10.1016/j.neunet.2018.07.011 | the systematic experiments over sampling ratios and the reported effects   | 2.7, 4.3      |
| Multi-task gradient conflict and gradient surgery as the alternative to gating       | Yu et al. 2020, PCGrad, NeurIPS                 | see section 9                | the gradient surgery formulation and the multi-task experiments            | 2.5, 4.3      |
| Dropout, the technique that was measured and removed                                 | Srivastava et al. 2014, *JMLR*                  | see section 9                | the dropout model and its regularisation effect                            | 4.3, 5.5      |

## 6. Combining predictions: NMS, ensembles, out-of-fold and stacking

| Claim or technique                                                                     | Paper                                              | DOI                           | Relevant section                                                                 | Used in       |
| -------------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------- | ------------- |
| Stacked generalisation, and cross-validation to build the level-1 training set         | Wolpert 1992, *Neural Networks*                    | 10.1016/S0893-6080(05)80023-1 | the stacked generalisation procedure and the construction of level-1 data        | 2.6, 4.6, 5.2 |
| Averaging over resampled models reduces variance                                       | Breiman 1996, Bagging, *Machine Learning*          | 10.1023/A:1018054314350       | the bagging procedure and the variance argument                                  | 2.6, 5.3      |
| Where stacking sits among ensemble methods                                             | Dietterich 2000, *MCS*                             | 10.1007/3-540-45014-9_1       | the taxonomy of ensemble methods                                                 | 2.6           |
| Train/test contamination invalidates an evaluation, and how it arises                  | Kaufman et al. 2012, *ACM TKDD*                    | 10.1145/2382577.2382579       | the leakage taxonomy and the contamination examples                              | 2.6, 5.9      |
| Leakage types and the recommendation to split on the independent unit                  | Kapoor and Narayanan 2023, *Patterns*              | 10.1016/j.patter.2023.100804  | the leakage-type taxonomy and the recommendations on dataset splitting           | 2.6, 3.3, 5.9 |
| Cross-validation estimates are biased unless the folds are independent                 | Varma and Simon 2006, *BMC Bioinformatics*         | 10.1186/1471-2105-7-91        | the comparison of error estimates across fold counts and nested cross-validation | 2.6, 3.3      |
| Soft-NMS as the rescoring alternative to greedy NMS                                    | Bodla et al. 2017, *ICCV*                          | 10.1109/ICCV.2017.593         | the linear and Gaussian rescoring, and the results                               | 2.6, 4.7      |
| Weighted box fusion and the comparison against NMS, Soft-NMS and non-maximum weighting | Solovyev et al. 2021, *Image and Vision Computing* | 10.1016/j.imavis.2021.104117  | the fusion algorithm and the comparison section                                  | 2.6, 5.5      |
| Self-attention and multi-head attention as the mechanism in the set stacker            | Vaswani et al. 2017, NeurIPS                       | see section 9                 | scaled dot-product attention and multi-head attention                            | 2.6, 4.6      |
| Set prediction with matching, which is the design pattern the set stacker follows      | Carion et al. 2020, DETR, *ECCV*                   | 10.1007/978-3-030-58452-8_13  | the set prediction loss and the transformer decoder                              | 2.6, 4.6      |

## 7. Evaluation metrics and validity

| Claim or technique                                                             | Paper                                      | DOI                              | Relevant section                                                                  | Used in            |
| ------------------------------------------------------------------------------ | ------------------------------------------ | -------------------------------- | --------------------------------------------------------------------------------- | ------------------ |
| IoU matching and the average precision definition                              | Everingham et al. 2010, PASCAL VOC, *IJCV* | 10.1007/s11263-009-0275-4        | the evaluation methodology defining IoU, precision, recall and AP                 | 2.6, 3.6, 5.1      |
| Per-category averaging, which is the macro mAP used here                       | Lin et al. 2014, COCO, *ECCV*              | 10.1007/978-3-319-10602-1_48     | the evaluation metrics and the per-category averaging                             | 3.6, 5.1, 5.4      |
| FROC as the sensitivity against false positives per image, and a summary index | Bandos et al. 2009, *Biometrics*           | 10.1111/j.1541-0420.2008.01049.x | the FROC curve definition and the summary index                                   | 2.6, 3.6, 5.1, 5.3 |
| Probability calibration, needed before a score can be called a risk            | Niculescu-Mizil and Caruana 2005, *ICML*   | 10.1145/1102351.1102430          | the histogram and sigmoid calibration methods, and the comparison across learners | 5.8, 6.3           |

## 8. Interpretability

| Claim or technique                                      | Paper                                   | DOI                  | Relevant section         | Used in  |
| ------------------------------------------------------- | --------------------------------------- | -------------------- | ------------------------ | -------- |
| Class activation style attribution for a CNN's decision | Selvaraju et al. 2017, Grad-CAM, *ICCV* | 10.1109/ICCV.2017.74 | the Grad-CAM formulation | 4.9, 5.6 |

## 9. No DOI: preprints, standards and software

Cite these with the identifier or publisher and an access date. They cannot go in the DOI
column, and Chapter 8 needs one consistent style for them.

| Item                                                                            | Reference form                                                   | Relevant part                                 | Used in       |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------- | ------------- |
| Fast scanning with max-pooling CNNs: dense evaluation instead of patch scanning | Giusti et al. 2013, arXiv:1302.1700                              | the method and the scanning discussion        | 2.4           |
| OverFeat: fully convolutional inference and localisation                        | Sermanet et al. 2014, arXiv:1312.6229                            | the dense inference and localisation sections | 2.4           |
| Batch normalisation                                                             | Ioffe and Szegedy 2015, arXiv:1502.03167                         | normalisation via mini-batch statistics       | 2.5, 4.3      |
| Adam                                                                            | Kingma and Ba 2015, arXiv:1412.6980                              | the algorithm                                 | 4.3           |
| Decoupled weight decay                                                          | Loshchilov and Hutter 2019, ICLR, arXiv:1711.05101               | the decoupled weight decay section            | 2.5, 4.3      |
| Cosine annealing with restarts                                                  | Loshchilov and Hutter 2017, ICLR, arXiv:1608.03983               | the cosine schedule                           | 2.5, 4.3      |
| Gradual warmup                                                                  | Goyal et al. 2017, arXiv:1706.02677                              | the gradual warmup section                    | 2.5, 4.3      |
| Attention                                                                       | Vaswani et al. 2017, NeurIPS, arXiv:1706.03762                   | scaled dot-product and multi-head attention   | 2.6, 4.6      |
| Gradient surgery for multi-task learning                                        | Yu et al. 2020, NeurIPS, arXiv:2001.06782                        | the gradient surgery formulation              | 2.5, 4.3      |
| Calibration of modern neural networks                                           | Guo et al. 2017, ICML, arXiv:1706.04599                          | temperature scaling and reliability diagrams  | 5.8           |
| Mosaic augmentation                                                             | Bochkovskiy et al. 2020, arXiv:2004.10934                        | the bag-of-freebies section                   | 2.5, 4.5      |
| YOLO26 architecture, as cited in the project brief                              | Ultralytics, arXiv:2509.25164                                    | the architectural enhancements section        | 2.4, 4.5      |
| Dropout                                                                         | Srivastava et al. 2014, *JMLR* 15:1929-1958                      | the dropout model                             | 4.3, 5.5      |
| DDSM, the source dataset behind CBIS-DDSM                                       | Heath et al. 1998, technical report, University of South Florida | the dataset description                       | 2.2, 3.4      |
| Breast screening programme standards and recall or sensitivity targets          | NHS Breast Screening Programme publications, grey literature     | the programme standards tables                | 2.1, 3.6, 6.3 |
| The NHS AI in breast screening research programme named in the brief            | NHS England announcement or trial protocol                       | the programme description and its aims        | 1.1, 6.3      |
| BI-RADS assessment categories, if the report discusses reader reporting         | ACR BI-RADS Atlas, current edition                               | the assessment category definitions           | 2.1           |
| Keras 3, torchvision, ultralytics YOLO, SciPy, scikit-learn                     | software citations with version and access date                  | the API or feature used                       | 3.5, 4.1      |

## 10. To verify before citing

These were wanted but the DOI could not be confirmed on 2026-09-28. Either find the correct
record or drop the claim.

| Item                                                                                 | Note                                                                                     |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| SpatialDropout2D (Tompson et al. 2015, CVPR)                                         | the DOI tried resolved to a different paper; only needed for the dropout negative result |
| Multi-task loss weighting by uncertainty (Kendall et al. 2018, CVPR)                 | the DOI tried resolved to a different paper; PCGrad covers the same point                |
| Cascade or IoU-based re-scoring in detection (for the learned re-ranking discussion) | no DOI verified; Cascade R-CNN and IoU-Net are the likely candidates                     |
| Bootstrap confidence intervals (Efron and Tibshirani)                                | needed if 5.8 reports intervals; cite the book edition                                   |
| YOLO26 review, *Computer Science and Information Technology* 2026                    | DOI 10.13189/csit.2026.140302 resolves, but prefer the arXiv preprint cited in the brief |
| Original FROC paper (Bunch et al. 1978)                                              | no DOI resolved; Bandos 2009 covers the method                                           |
