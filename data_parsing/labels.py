"""
This file stores the label information for PAMAP2 and OPPORTUNITY

  Opportunity: milli g = 0.001 g
  PAMAP2 ms^-2 = 0.1 g
"""

keys = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24, 0]
labels = [
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "Nordic walking ",
    "watching TV ",
    "computer work ",
    "car driving ",
    "ascending stairs ",
    "descending stairs",
    "vacuum cleaning ",
    "ironing",
    "folding laundry",
    "house cleaning",
    "playing soccer",
    "rope jumping",
    "other (transient activities)",
]
pamap2_labels = dict(zip(keys, labels))

keys = [1, 2, 4, 5]
labels = ["stand", "walk", "sit", "lie"]
oppo_ori_labels = dict(zip(keys, labels))

keys = [1, 2, 4, 3]
labels = ["stand", "walk", "sit", "lie"]
oppo_labels = dict(zip(keys, labels))
