# Notes for final report

## Uncategorised notes
- The classification is binary, malignant:1, benign:0
- 

## Standardised Metrics
### Training terms (weighted Keras losses, summed per anchor population)
| Column | Meaning                                                                                |
| ------ | -------------------------------------------------------------------------------------- |
| Loss   | Total weighted training loss = obj + box + cls (all weights are 1.0).                  |
| obj    | Objectness loss, MaskedBinaryCrossentropy. Normalised by the count of sampled anchors. |
| box    | Box regression loss, MaskedHuber on the anchor deltas                                  |
| cls    | MaskedClassCrossentropy on positive anchors                                            |

### Validation metrics (recomputed from scratch every epoch by validate)
| Column | Meaning                                                                            |
| ------ | ---------------------------------------------------------------------------------- |
| AP50   | Average precision at IoU 0.5. This is what selects Saved best model. Ignores class |
| mAP50  | The macro average of per-class AP50 (ap50_macro)                                   |
| P      | Precision at the model's operating point                                           |
| R      | Recall at the model's operating point                                              |
| F1     | F1 score model's operating point                                                   |
| Det    | Mean post-NMS boxes per image at model's operating point                           |
