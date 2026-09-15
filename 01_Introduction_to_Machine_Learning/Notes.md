# ML Zoomcamp — Chapter 1: Introduction to Machine Learning

---

## 1.1 Introduction to Machine Learning

📺 [Video](https://www.youtube.com/watch?v=Crm_5n4mvmg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=2) | 🖼️ [Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-11-introduction-to-machine-learning)

![Introduction to ML](https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/main/01-intro/images/thumbnail-1-01.jpg)

### Core idea

The lesson explains ML using a **car price prediction** example.

- The model is trained on historical data made up of two parts:
  - **Features** — information about the object (e.g. `year`, `mileage`, `make`, `condition`, etc.)
  - **Target** — the property we want to predict (e.g. `price`)
- During training, the model **extracts patterns** from the features that explain the target.
- Once trained, the model is given **new data without the target**, and it uses the learned patterns to **predict** the target value.

> **ML = the process of extracting patterns from data**
> - **features** → input info about an object
> - **target** → the thing we want to predict for unseen objects

### Simple mental model

```
Training phase:
   [ Features + Target ]  --->  Model learns patterns

Prediction phase:
   [ Features only ]  --->  Trained Model  --->  Predicted Target
```

**Example — Car price prediction**

| year | mileage | make    | ... | price (target) |
|------|---------|---------|-----|-----------------|
| 2015 | 60,000  | Toyota  | ... | $12,000         |
| 2018 | 30,000  | Honda   | ... | $16,500         |
| 2020 | 10,000  | Ford    | ... | ?  ← model predicts this |

- Rows with a known `price` → used to **train** the model.
- New row without `price` → fed to the trained model to **predict** the price.

### Key takeaway

Machine Learning isn't about hard-coding rules — it's about letting a model **learn patterns from historical examples (features → target)** so it can make predictions on new, unseen data.

---

<!-- Next lesson notes (1.2 ML vs Rule-Based Systems) go below -->

