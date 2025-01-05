rtrec: Realtime Recommendation Library in Python
================================================

[![PyPI version](https://img.shields.io/pypi/v/rtrec.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/rtrec/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/rtrec.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/rtrec/)
[![CI status](https://github.com/myui/rtrec/actions/workflows/ci.yml/badge.svg)](https://github.com/myui/rtrec/actions)
[![Licence](https://img.shields.io/github/license/myui/rtrec.svg)](LICENSE.txt)

A realtime recommendation system supporting online updates.

## Highlights

- ❇️ Supporting online updates.
- ⚡️ Fast implementation (>=190k samples/sec training on laptop).
- ◍ efficient sparse data support.
- 🕑 decaying weights of user-item interactions based on recency.
- ![Rust](https://avatars.githubusercontent.com/u/5430905?s=20&v=4) experimental [Rust implementation](https://github.com/myui/rtrec/tree/rust)

## Supported Recommendation Algorithims

- Sparse [SLIM](https://ieeexplore.ieee.org/document/6137254) with [time-weighted](https://dl.acm.org/doi/10.1145/1099554.1099689) interactions.
- [Factorization Machines](https://ieeexplore.ieee.org/document/5694074) using [LightFM](https://github.com/lyst/lightfm)

## Installation

```bash
pip install rtrec
```

## Usage

Find usages in [notebooks](https://github.com/myui/rtrec/tree/main/notebooks)/[examples](https://github.com/myui/rtrec/tree/main/examples).

### Examples using Raw-level APIs

```py
# Dataset consists of user, item, tstamp, rating
import time
current_unixtime = time.time()
interactions = [('user_1', 'item_1', current_unixtime, 5.0),
                ('user_2', 'item_2', current_unixtime, -2.0),
                ('user_2', 'item_1', current_unixtime, 3.0),
                ('user_2', 'item_4', current_unixtime, 3.0),
                ('user_1', 'item_3', current_unixtime, 4.0)]

# Fit SLIM model
from rtrec.models import SLIM
model = SLIM()
model.fit(interactions)

# can fit from streams using yield as follows:
def yield_interactions():
    for interaction in interactions:
        yield interaction
model.fit(yield_interactions())

# Recommend top-5 items for a user
recommendations = model.recommend('user_1', top_k=5)
assert recommendations == ["item_4", "item_2"]
```

### Examples using high level DataFrame APIs

```py
# load dataset
from rtrec.experiments.datasets import load_dataset
df = load_dataset(name='movielens_1m')

# Split data set by temporal user split
from rtrec.experiments.split import temporal_user_split
train_df, test_df = temporal_user_split(df)

# Initialize SLIM model with custom options
from rtrec.recommender import Recommender
from rtrec.models import SLIM
model = SLIM(min_value=0, max_value=15, decay_in_days=180, nn_feature_selection=50)
recommender = Recommender(model)

# Bulk fit
recommender.bulk_fit(train_df)

# Partial fit
from rtrec.experiments.split import temporal_split
test_df1, test_df2 = temporal_split(test_df, test_frac=0.5)

recommender.fit(test_df1, update_interaction=True, parallel=True)

# Evaluation
metrics = recommender.evaluate(test_df2, recommend_size=10, filter_interacted=True)
print(metrics)

# User to Item Recommendation
recommended = recommender.recommend(user=10, top_k=10, filter_interacted=True)
assert len(recommended) == 10

# Item to Item recommendation
similar_items = recommender.similar_items(query_items=[3,10], top_k=5)
assert len(similar_items) == 2
```