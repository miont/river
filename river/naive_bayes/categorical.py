import collections
from typing import List, Union, Iterable, Dict
import math

from scipy import special

from river import base

__all__ = ["CategoricalNB"]

eps = 10e-10
SampleType = Union[Iterable[str], Iterable[int]]
ClassType = Union[str, int]
FeatureType = Union[str, int]


class CategoricalNB(base.Classifier):
    """Naive Bayes classifier for categorically distributed data.

    The categorical Naive Bayes classifier is suitable for classification with
    discrete features that are categorically distributed. The input vector
    can contain names of categories as strings or integer numbers
    `0, …, n - 1`, where `n` refers to the total number of categories
    for the given feature.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

    Attributes
    ----------
    feature_counts : collections.defaultdict
        Total frequencies per feature and class.
    class_totals : collections.Counter
        Total frequencies per class.

    Examples
    --------


    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = None

    def _init_feature_counts(self, n_features: int) -> None:
        self.feature_counts = [
            collections.defaultdict(collections.Counter) for _ in range(n_features)
        ]

    @property
    def n_classes(self) -> int:
        return len(self.class_counts)

    @property
    def n_categories(self) -> List[int]:
        return [len(cats) for cats in self.feature_counts]

    @property
    def classes_(self) -> List[ClassType]:
        return list(self.class_counts.keys())

    def learn_one(self, x: SampleType, y: ClassType) -> "CategoricalNB":
        """Updates the model with a single observation.

        Parameters
        ----------
        x
            Vector of categorical feature values.
        y
            Target class.

        Returns
        -------
        self

        """
        self.class_counts.update((y,))

        if not self.feature_counts:
            self._init_feature_counts(len(x))
        else:
            if len(x) != len(self.feature_counts):
                raise Exception(
                    f"Features count in the given sample differs "
                    + f"from previous ones: {len(x)} != {len(self.feature_counts)}"
                )

        for i, f in enumerate(x):
            self.feature_counts[i][f].update((y,))

        return self

    def p_feature_given_class(self, idx: int, f: FeatureType, c: ClassType) -> float:
        num = self.feature_counts[idx].get(f, {}).get(c, 0) + self.alpha
        den = self.class_counts[c] + self.alpha * self.n_categories[idx]
        return num / den

    def p_class(self, c: ClassType) -> float:
        return self.class_counts[c] / sum(self.class_counts.values())

    def joint_log_likelihood(self, x) -> Dict[ClassType, float]:
        """Computes the joint log likelihood of input features.

        Parameters
        ----------
        x
            Vector of categorical feature values.

        Returns
        -------
        Mapping between classes and joint log likelihood.

        """
        return {
            c: math.log(self.p_class(c))
            + sum(math.log(eps + self.p_feature_given_class(i, f, c)) for i, f in enumerate(x))
            for c in self.classes_
        }

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jll = self.joint_log_likelihood(x)
        if not jll:
            return {}
        lse = special.logsumexp(list(jll.values()))
        return {label: math.exp(ll - lse) for label, ll in jll.items()}

    @property
    def _multiclass(self):
        return True
