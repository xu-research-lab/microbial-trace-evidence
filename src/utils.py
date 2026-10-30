import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RF_PARAM_GRID = {
    "max_depth": [None],
    "max_features": np.logspace(1, 6, num=6, base=2, dtype=int),
    "min_samples_split": [2, 3, 7],
    "min_samples_leaf": [1, 3, 7],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}

MAT_COLOR_DICT = {
    'plastic': '#1f77b4',
    'glass': '#2ca02c',
    'ceramic': '#d62728',
    'metal': '#9467bd',
    'wood': '#8c564b',
}


# ---------------------------------------------------------------------------
# Statistical utility functions
# ---------------------------------------------------------------------------

def jackknife_se(accs):
    """Compute Jackknife standard error.

    Parameters
    ----------
    accs : numpy.ndarray
        1-D array of accuracy values from jackknife folds.

    Returns
    -------
    float
        Jackknife standard error. Returns 0.0 when len(accs) <= 1.
    """
    n = len(accs)
    if n <= 1:
        return 0.0
    return np.sqrt((n - 1) / n * np.sum((accs - accs.mean()) ** 2))


def get_significance_stars(p, non_sig=''):
    """Return significance stars based on p-value.

    Parameters
    ----------
    p : float
        p-value (non-negative).
    non_sig : str, optional
        Label for non-significant results (default ''). Use 'n.s.' or ' ' for
        plots that need an explicit non-significant marker.

    Returns
    -------
    str
        '***' if p < 0.001, '**' if p < 0.01, '*' if p < 0.05,
        non_sig otherwise.
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return non_sig


def jackknife_auc_unpaired_test(fold_matrix_A, fold_matrix_B,
                                auc_divisor='max_N', alternative='two-sided'):
    """Perform unpaired Jackknife AUC test on independent fold matrices.

    Parameters
    ----------
    fold_matrix_A : numpy.ndarray
        Shape (max_N, n_folds_A). Top-N accuracy matrix for group A.
    fold_matrix_B : numpy.ndarray
        Shape (max_N, n_folds_B). Top-N accuracy matrix for group B.
    auc_divisor : str, optional
        'max_N': divide AUC by max_N (default, used in part1/4/5).
        'max_N_minus_1': divide AUC by (max_N - 1) (used in part3).
    alternative : str, optional
        'two-sided': two-sided test (default, used in part1/3/5).
        'greater': one-sided test H_A > H_B (used in part4).
        'less': one-sided test H_A < H_B.

    Returns
    -------
    float
        p-value in [0, 1]. Returns 1.0 when se_diff == 0 or n <= 1.
    """
    max_N = fold_matrix_A.shape[0]

    if auc_divisor == 'max_N_minus_1':
        divisor = max_N - 1
    else:
        divisor = max_N

    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc_A = _trapz(fold_matrix_A, axis=0) / divisor
    auc_B = _trapz(fold_matrix_B, axis=0) / divisor

    mean_A = auc_A.mean()
    mean_B = auc_B.mean()
    n_A = len(auc_A)
    n_B = len(auc_B)

    if n_A <= 1 or n_B <= 1:
        return 1.0

    se_A = np.sqrt((n_A - 1) / n_A * np.sum((auc_A - mean_A) ** 2))
    se_B = np.sqrt((n_B - 1) / n_B * np.sum((auc_B - mean_B) ** 2))

    mean_diff = mean_A - mean_B
    se_diff = np.sqrt(se_A**2 + se_B**2)

    if se_diff == 0:
        return 1.0

    z_stat = mean_diff / se_diff

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return p_value


def compute_jackknife_topn_stats(fold_matrix, auc_divisor='max_N'):
    """Compute mean, SE, AUC, and AUC SE from a jackknife fold matrix.

    Parameters
    ----------
    fold_matrix : numpy.ndarray
        Shape (max_N, n_folds). Top-N accuracy values across jackknife folds.
    auc_divisor : str, optional
        'max_N': divide AUC by max_N (default).
        'max_N_minus_1': divide AUC by (max_N - 1).

    Returns
    -------
    dict
        Keys: 'means' (ndarray, shape (max_N,)),
              'se' (ndarray, shape (max_N,)),
              'auc' (float),
              'se_auc' (float).
    """
    max_N = fold_matrix.shape[0]
    means = fold_matrix.mean(axis=1)
    se = np.array([jackknife_se(fold_matrix[N, :]) for N in range(max_N)])

    divisor = max_N - 1 if auc_divisor == 'max_N_minus_1' else max_N
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc_folds = _trapz(fold_matrix, axis=0) / divisor
    auc = auc_folds.mean()
    se_auc = jackknife_se(auc_folds)

    return {'means': means, 'se': se, 'auc': auc, 'se_auc': se_auc}


def extract_masked_fold_matrix(fold_tensor, sample_mask, *group_indices):
    """Extract a 2-D jackknife fold matrix from a higher-rank fold tensor.

    Parameters
    ----------
    fold_tensor : numpy.ndarray
        Array whose first axis is Top-N and second axis is sample/fold. Any
        remaining axes correspond to grouping dimensions such as material,
        touch order, year, or role.
    sample_mask : array-like of bool
        Boolean mask selecting the valid samples/folds along axis 1.
    *group_indices : int
        Indices selecting a single slice from each grouping axis after axis 1.

    Returns
    -------
    numpy.ndarray
        Shape (max_N, n_selected_folds).
    """
    sample_mask = np.asarray(sample_mask, dtype=bool)
    indexer = (slice(None), sample_mask, *group_indices)
    return np.asarray(fold_tensor[indexer], dtype=float)


def compute_auc_summary(fold_matrix, auc_divisor='max_N'):
    """Summarize normalized AUC values across jackknife folds.

    Parameters
    ----------
    fold_matrix : numpy.ndarray
        Shape (max_N, n_folds) or (max_N,). Top-N accuracy values.
    auc_divisor : str, optional
        'max_N' or 'max_N_minus_1'. Matches ``compute_jackknife_topn_stats``.

    Returns
    -------
    dict
        Keys: 'auc_vals' (ndarray), 'auc' (float), 'error' (float), 'n' (int).
    """
    fold_matrix = np.asarray(fold_matrix, dtype=float)
    if fold_matrix.ndim == 1:
        fold_matrix = fold_matrix[:, np.newaxis]
    if fold_matrix.ndim != 2:
        raise ValueError('fold_matrix must be 1-D or 2-D.')

    n_folds = fold_matrix.shape[1]
    if n_folds == 0:
        return {
            'auc_vals': np.array([], dtype=float),
            'auc': 0.0,
            'error': 0.0,
            'n': 0,
        }

    max_N = fold_matrix.shape[0]
    divisor = max_N - 1 if auc_divisor == 'max_N_minus_1' else max_N
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc_vals = _trapz(fold_matrix, axis=0) / divisor
    auc_mean = float(auc_vals.mean())
    auc_error = float(jackknife_se(auc_vals))

    return {
        'auc_vals': auc_vals,
        'auc': auc_mean,
        'error': auc_error,
        'n': int(n_folds),
    }


def compute_masked_jackknife_topn_stats(fold_tensor, sample_mask, *group_indices,
                                        auc_divisor='max_N'):
    """Compute Top-N jackknife stats for a masked group slice of a fold tensor.

    Parameters
    ----------
    fold_tensor : numpy.ndarray
        Array whose first axis is Top-N and second axis is sample/fold.
    sample_mask : array-like of bool
        Boolean mask selecting valid samples/folds along axis 1.
    *group_indices : int
        Indices selecting a single slice from each grouping axis after axis 1.
    auc_divisor : str, optional
        'max_N' or 'max_N_minus_1'.

    Returns
    -------
    dict
        ``compute_jackknife_topn_stats`` result plus:
        'fold_matrix' (ndarray) and 'n' (int).
    """
    fold_matrix = extract_masked_fold_matrix(fold_tensor, sample_mask, *group_indices)
    stats_dict = compute_jackknife_topn_stats(fold_matrix, auc_divisor=auc_divisor)
    stats_dict['fold_matrix'] = fold_matrix
    stats_dict['n'] = int(fold_matrix.shape[1])
    return stats_dict


def compute_jackknife_fold_tensor(model, X, sample_metadata, class_labels,
                                  group_specs, max_N=None, progress_every=0):
    """Compute jackknife fold tensors for one or more categorical groupings.

    Parameters
    ----------
    model : estimator
        Must expose ``predict_proba`` and ``classes_``.
    X : array-like
        Feature matrix aligned to ``sample_metadata``.
    sample_metadata : pandas.DataFrame
        Metadata aligned to ``X``.
    class_labels : sequence
        Ordered class labels used to rank ``predict_proba`` outputs.
    group_specs : sequence of dict
        Each dict must provide:
        - ``name``: output key
        - ``order``: ordered category labels
        - ``column``: metadata column name
        Optional:
        - ``value_map``: callable mapping metadata value -> group label
        - ``max_N``: override for this group
    max_N : int, optional
        Default Top-N depth. Falls back to ``len(class_labels)``.
    progress_every : int, optional
        Print progress every N left-out folds when > 0.

    Returns
    -------
    dict
        ``{name: {'jk_bar': ..., 'jk_topn': ..., 'order': ..., 'denominators': ...}}``
        plus ``'max_N'`` and ``'n_test'``.
    """
    if max_N is None:
        max_N = len(class_labels)

    X_test = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    n_test = X_test.shape[0]
    results = {'max_N': max_N, 'n_test': n_test}

    group_state = []
    for spec in group_specs:
        order = list(spec['order'])
        group_state.append({
            'name': spec['name'],
            'order': order,
            'column': spec['column'],
            'value_map': spec.get('value_map'),
            'max_N': spec.get('max_N', max_N),
            'to_i': {value: i for i, value in enumerate(order)},
            'jk_bar': np.zeros((spec.get('max_N', max_N), n_test, len(order))),
            'fold_denominators': np.zeros((n_test, len(order))),
        })

    for fold_i in range(n_test):
        if progress_every and (fold_i + 1) % progress_every == 0:
            print(f"   Progress: {fold_i+1}/{n_test}")

        keep_idx = [j for j in range(n_test) if j != fold_i]
        probs_subset = model.predict_proba(X_test[keep_idx])
        ranks_subset = pd.DataFrame(probs_subset, columns=class_labels).rank(axis=1, ascending=False)

        for idx_in_subset, orig_idx in enumerate(keep_idx):
            sid = sample_metadata.index[orig_idx]
            md = sample_metadata.loc[sid]
            true_ind = md['individual']
            rank = ranks_subset.iloc[idx_in_subset][true_ind]

            for state in group_state:
                value = md[state['column']]
                if state['value_map'] is not None:
                    value = state['value_map'](value)
                g_i = state['to_i'].get(value, -1)
                if g_i < 0:
                    continue

                state['fold_denominators'][fold_i, g_i] += 1
                for N in range(state['max_N']):
                    if rank <= (N + 1):
                        state['jk_bar'][N, fold_i, g_i] += 1

    for state in group_state:
        for N in range(state['max_N']):
            state['jk_bar'][N] = np.divide(
                state['jk_bar'][N],
                state['fold_denominators'],
                out=np.zeros_like(state['jk_bar'][N]),
                where=state['fold_denominators'] != 0,
            )

        results[state['name']] = {
            'jk_bar': state['jk_bar'],
            'jk_topn': state['jk_bar'].copy(),
            'order': state['order'],
            'denominators': state['fold_denominators'],
        }

    return results


def compute_jackknife_tensor(model, X, sample_metadata, contribution_getter,
                             class_labels=None, max_N=None, progress_every=0,
                             tensor_shape=None, normalize=True,
                             return_mean_probs=False):
    """Compute a jackknife Top-N tensor from generic per-sample contributions.

    Parameters
    ----------
    model : estimator
        Must expose ``predict_proba`` and ``classes_``.
    X : array-like
        Feature matrix aligned to ``sample_metadata``.
    sample_metadata : pandas.DataFrame
        Metadata aligned to ``X``.
    contribution_getter : callable
        Called once per kept sample as
        ``contribution_getter(md=..., orig_idx=..., idx_in_subset=...,
        fold_i=..., keep_idx=..., subset_meta=..., probs_subset=...,
        ranks_subset=..., classes=..., sample_metadata=...)``.
        It must return an iterable of dicts. Each dict must contain:
        - ``group_index``: int or tuple of ints selecting the output slice
        - ``target``: class label whose rank should be evaluated
        Optional:
        - ``score_weight``: value added when ``rank <= N + 1`` (default 1.0)
        - ``denom_weight``: value added to the fold denominator (default 1.0)
    class_labels : sequence, optional
        Ordered class labels. Defaults to ``model.classes_``.
    max_N : int, optional
        Top-N depth. Defaults to ``len(class_labels)``.
    progress_every : int, optional
        Print progress every N left-out folds when > 0.
    tensor_shape : tuple, optional
        Grouping shape after the ``(max_N, n_test)`` axes. Required.
    normalize : bool, optional
        If True, divide by per-fold denominators after accumulation.
    return_mean_probs : bool, optional
        If True, also return mean predicted probabilities across folds.

    Returns
    -------
    dict
        Keys include ``jk_bar``, ``jk_topn``, ``fold_denominators``,
        ``max_N``, ``n_test`` and optionally ``mean_probs``.
    """
    if tensor_shape is None:
        raise ValueError('tensor_shape is required.')

    if class_labels is None:
        class_labels = model.classes_
    if max_N is None:
        max_N = len(class_labels)

    X_test = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    n_test = X_test.shape[0]
    jk_bar = np.zeros((max_N, n_test, *tensor_shape))
    fold_denominators = np.zeros((n_test, *tensor_shape))

    prob_accumulators = None
    sample_counts = None
    if return_mean_probs:
        prob_accumulators = np.zeros((n_test, len(class_labels)))
        sample_counts = np.zeros(n_test)

    for fold_i in range(n_test):
        if progress_every and (fold_i + 1) % progress_every == 0:
            print(f"   Progress: {fold_i+1}/{n_test}")

        keep_idx = [j for j in range(n_test) if j != fold_i]
        subset_meta = sample_metadata.iloc[keep_idx]
        probs_subset = model.predict_proba(X_test[keep_idx])
        ranks_subset = pd.DataFrame(probs_subset, columns=class_labels).rank(
            axis=1, ascending=False
        )

        for idx_in_subset, orig_idx in enumerate(keep_idx):
            if return_mean_probs:
                prob_accumulators[orig_idx] += probs_subset[idx_in_subset]
                sample_counts[orig_idx] += 1

            md = sample_metadata.iloc[orig_idx]
            contributions = contribution_getter(
                md=md,
                orig_idx=orig_idx,
                idx_in_subset=idx_in_subset,
                fold_i=fold_i,
                keep_idx=keep_idx,
                subset_meta=subset_meta,
                probs_subset=probs_subset,
                ranks_subset=ranks_subset,
                classes=class_labels,
                sample_metadata=sample_metadata,
            )

            for contrib in contributions:
                target = contrib['target']
                if target not in class_labels:
                    continue

                group_index = contrib['group_index']
                if not isinstance(group_index, tuple):
                    group_index = (group_index,)

                score_weight = contrib.get('score_weight', 1.0)
                denom_weight = contrib.get('denom_weight', 1.0)
                rank = ranks_subset.iloc[idx_in_subset][target]

                fold_denominators[(fold_i, *group_index)] += denom_weight
                for N in range(max_N):
                    if rank <= (N + 1):
                        jk_bar[(N, fold_i, *group_index)] += score_weight

    if normalize:
        for N in range(max_N):
            jk_bar[N] = np.divide(
                jk_bar[N],
                fold_denominators,
                out=np.zeros_like(jk_bar[N]),
                where=fold_denominators != 0,
            )

    results = {
        'jk_bar': jk_bar,
        'jk_topn': jk_bar.copy(),
        'fold_denominators': fold_denominators,
        'max_N': max_N,
        'n_test': n_test,
    }

    if return_mean_probs:
        results['mean_probs'] = np.divide(
            prob_accumulators,
            sample_counts[:, None],
            out=np.zeros_like(prob_accumulators),
            where=sample_counts[:, None] != 0,
        )
        results['sample_counts'] = sample_counts

    return results


def compute_cached_jackknife_tensor(
    model,
    X,
    sample_metadata,
    contribution_getter,
    tensor_shape,
    cache_path=None,
    class_labels=None,
    max_N=None,
    progress_every=0,
    normalize=True,
    return_mean_probs=False,
    bar_key='jk_bar',
    topn_key='jk_topn',
    topn_getter=None,
    cache_extras=None,
    force_refresh=False,
    load_message="Loading cached jackknife results...",
    run_message="Running jackknife on test set...",
    save_message="Jackknife results saved.",
):
    """Load or compute a cached jackknife tensor with notebook-friendly keys.

    This wraps the repeated notebook pattern:
    load cache -> compute tensor when missing -> mirror ``jk_bar`` into a
    custom result dict -> save cache -> expose ``max_N`` from the tensor.

    Parameters
    ----------
    model, X, sample_metadata, contribution_getter, tensor_shape :
        Passed through to :func:`compute_jackknife_tensor`.
    cache_path : str, optional
        Cache file to load/save. When omitted, always compute in memory.
    class_labels, max_N, progress_every, normalize, return_mean_probs :
        Passed through to :func:`compute_jackknife_tensor`.
    bar_key, topn_key : str, optional
        Keys used when persisting the cached tensor payload.
    topn_getter : callable, optional
        Called as ``topn_getter(jk_result)`` after computation to derive the
        cached ``topn_key`` payload. Defaults to ``jk_result['jk_topn']``.
    cache_extras : dict, optional
        Extra metadata to include in the returned payload and cache.
    force_refresh : bool, optional
        If True, recompute even when ``cache_path`` already exists.
    load_message, run_message, save_message : str, optional
        Status messages printed during execution.

    Returns
    -------
    dict
        Cached payload plus canonical aliases: ``jk_bar``, ``jk_topn``,
        and ``max_N``.
    """
    cache_extras = {} if cache_extras is None else dict(cache_extras)

    if cache_path and os.path.exists(cache_path) and not force_refresh:
        print(load_message)
        payload = joblib.load(cache_path)
        payload['jk_bar'] = payload[bar_key]
        payload['jk_topn'] = payload.get(topn_key, payload['jk_bar'].copy())
        payload['max_N'] = payload['jk_bar'].shape[0]
        return payload

    print(run_message)
    jk_result = compute_jackknife_tensor(
        model=model,
        X=X,
        sample_metadata=sample_metadata,
        contribution_getter=contribution_getter,
        class_labels=class_labels,
        max_N=max_N,
        progress_every=progress_every,
        tensor_shape=tensor_shape,
        normalize=normalize,
        return_mean_probs=return_mean_probs,
    )

    topn_value = (
        topn_getter(jk_result) if topn_getter is not None else jk_result['jk_topn']
    )

    payload = {
        bar_key: jk_result['jk_bar'],
        topn_key: topn_value,
        **cache_extras,
    }

    if return_mean_probs:
        payload['mean_probs'] = jk_result['mean_probs']
        payload['sample_counts'] = jk_result['sample_counts']

    payload['jk_bar'] = jk_result['jk_bar']
    payload['jk_topn'] = topn_value
    payload['max_N'] = jk_result['max_N']

    if cache_path:
        cache_payload = {
            key: value
            for key, value in payload.items()
            if key not in {'jk_bar', 'jk_topn', 'max_N'}
        }
        joblib.dump(cache_payload, cache_path)
        print(save_message)

    return payload


def compute_persistence_jackknife(
    model,
    test_exp,
    cache_path=None,
    max_N=20,
    progress_every=50,
    exp1_times=('Immediately', '5min', '1hr', '18hr'),
    exp2_times=('Immediately', '24hr', '48hr', '72hr'),
    materials=('plastic', 'ceramic'),
    experiments=('exp201603', 'exp201709'),
):
    """Compute or load the persistence jackknife tensor used in part 3.

    Parameters
    ----------
    model : estimator
        Must expose ``classes_`` and ``predict_proba``.
    test_exp : experiment-like object
        Must expose ``data``, ``shape``, and ``sample_metadata``.
    cache_path : str, optional
        If provided and the file exists, load cached results from this path.
        Otherwise compute and optionally write the cache.
    max_N : int, optional
        Top-N depth.
    progress_every : int, optional
        Print progress every N folds.
    exp1_times, exp2_times, materials, experiments : sequence, optional
        Persistence grouping definitions matching the notebook analysis.

    Returns
    -------
    dict
        Keys include ``jk4_acc``, ``materials``, ``exp1_times``,
        ``exp2_times``, ``experiments``, and ``max_N``.
    """
    if cache_path and os.path.exists(cache_path):
        print("Loading cached jackknife results from:", cache_path)
        results = joblib.load(cache_path)
        results['max_N'] = results['jk4_acc'].shape[0]
        return results

    print("Running jackknife on test set... This might take a while.")

    sample_metadata = test_exp.sample_metadata
    classes = model.classes_
    materials = list(materials)
    exp1_times = list(exp1_times)
    exp2_times = list(exp2_times)
    experiments = list(experiments)

    time_to_sec = {
        'Immediately': 1,
        '5min': 300,
        '1hr': 3600,
        '18hr': 64800,
        '24hr': 86400,
        '48hr': 172800,
        '72hr': 259200,
    }
    sec_to_label = {v: k for k, v in time_to_sec.items()}
    exp_to_i = {exp_name: i for i, exp_name in enumerate(experiments)}
    mat_to_i = {mat_name: i for i, mat_name in enumerate(materials)}

    def persistence_jk_contrib(md, **ctx):
        exp_i = exp_to_i.get(md['experiment'], -1)
        mat_i = mat_to_i.get(md['surface_type'], -1)
        if exp_i < 0 or mat_i < 0:
            return []

        t_labels = exp1_times if exp_i == 0 else exp2_times
        t_label = sec_to_label.get(int(md['persistence_time_sec']))
        if t_label not in t_labels:
            return []

        return [{
            'group_index': (exp_i, mat_i, t_labels.index(t_label)),
            'target': md['individual'],
            'denom_weight': 0.0,
        }]

    jk_result = compute_jackknife_tensor(
        model=model,
        X=test_exp.data,
        sample_metadata=sample_metadata,
        contribution_getter=persistence_jk_contrib,
        class_labels=classes,
        max_N=max_N,
        progress_every=progress_every,
        tensor_shape=(len(experiments), len(materials), max(len(exp1_times), len(exp2_times))),
        normalize=False,
    )

    jk4_acc = jk_result['jk_bar']
    n_test = test_exp.shape[0]

    print("Normalizing results...")
    for fold_i in range(n_test):
        md_leftout = sample_metadata.iloc[fold_i]
        lo_exp = md_leftout['experiment']
        lo_mat = md_leftout['surface_type']
        lo_time = int(md_leftout['persistence_time_sec'])

        for exp_i, curr_exp_name in enumerate(experiments):
            t_labels = exp1_times if exp_i == 0 else exp2_times
            for mat_i, mat_name in enumerate(materials):
                for time_j, t_label in enumerate(t_labels):
                    t_sec = time_to_sec[t_label]
                    mask = (
                        (sample_metadata['experiment'] == curr_exp_name) &
                        (sample_metadata['surface_type'] == mat_name) &
                        (sample_metadata['persistence_time_sec'] == t_sec)
                    )
                    total = mask.sum()
                    if total == 0:
                        continue

                    is_left_out = (
                        (lo_exp == curr_exp_name) and
                        (lo_mat == mat_name) and
                        (lo_time == t_sec)
                    )
                    denom = total - 1 if is_left_out else total

                    if denom > 0:
                        jk4_acc[:, fold_i, exp_i, mat_i, time_j] /= denom
                    else:
                        jk4_acc[:, fold_i, exp_i, mat_i, time_j] = np.nan

    results = {
        'jk4_acc': jk4_acc,
        'materials': materials,
        'exp1_times': exp1_times,
        'exp2_times': exp2_times,
        'experiments': experiments,
        'max_N': max_N,
    }

    if cache_path:
        joblib.dump(results, cache_path)
        print("Jackknife results successfully calculated and cached.")

    return results


def compute_cross_year_jackknife(
    model,
    test_exp,
    cache_path=None,
    progress_every=50,
    types_list=('skin', 'plastic', 'glass', 'metal', 'wood', 'ceramic'),
):
    """Compute or load the cross-year jackknife payload used in part 5.

    Parameters
    ----------
    model : estimator
        Must expose ``classes_`` and ``predict_proba``.
    test_exp : experiment-like object
        Must expose ``data`` and ``sample_metadata``.
    cache_path : str, optional
        If provided and the file exists, load cached results from this path.
        Otherwise compute and optionally write the cache.
    progress_every : int, optional
        Print progress every N folds.
    types_list : sequence, optional
        Ordered material groups for the cross-material analysis.

    Returns
    -------
    dict
        Keys include ``yprob_jk_mean``, ``jk_bar_year``, ``jk_bar_type``,
        ``years_list``, ``types_list``, ``year_to_i``, ``type_to_i``,
        and ``max_N``.
    """
    if cache_path and os.path.exists(cache_path):
        print("Loading cached unified jackknife results...")
        results = joblib.load(cache_path)
        results['year_to_i'] = {y: i for i, y in enumerate(results['years_list'])}
        results['type_to_i'] = {t: i for i, t in enumerate(results['types_list'])}
        results['max_N'] = results['jk_bar_year'].shape[0]
        return results

    print("Running unified jackknife on test set...")

    sample_metadata = test_exp.sample_metadata
    classes = model.classes_
    max_N = len(classes)

    skin_mask = sample_metadata['sample_type'] == 'skin'
    years_list = sorted([
        int(y) for y in sample_metadata[skin_mask]['year'].dropna().unique()
    ])
    year_to_i = {y: i for i, y in enumerate(years_list)}

    types_list = list(types_list)
    type_to_i = {t: i for i, t in enumerate(types_list)}

    def year_jk_contrib(md, classes, **ctx):
        if md['individual'] not in classes:
            return []
        if md['sample_type'] != 'skin' or pd.isna(md['year']):
            return []
        y_val = int(md['year'])
        if y_val not in year_to_i:
            return []
        return [{
            'group_index': year_to_i[y_val],
            'target': md['individual'],
        }]

    def type_jk_contrib(md, classes, **ctx):
        if md['individual'] not in classes:
            return []
        stype = md['surface_type']
        if stype in ['Lpalm', 'Rpalm', 'skin']:
            stype = 'skin'
        if stype not in type_to_i:
            return []
        return [{
            'group_index': type_to_i[stype],
            'target': md['individual'],
        }]

    year_result = compute_jackknife_tensor(
        model=model,
        X=test_exp.data,
        sample_metadata=sample_metadata,
        contribution_getter=year_jk_contrib,
        class_labels=classes,
        max_N=max_N,
        progress_every=progress_every,
        tensor_shape=(len(years_list),),
        return_mean_probs=True,
    )

    type_result = compute_jackknife_tensor(
        model=model,
        X=test_exp.data,
        sample_metadata=sample_metadata,
        contribution_getter=type_jk_contrib,
        class_labels=classes,
        max_N=max_N,
        progress_every=progress_every,
        tensor_shape=(len(types_list),),
    )

    results = {
        'yprob_jk_mean': pd.DataFrame(
            year_result['mean_probs'],
            index=sample_metadata.index,
            columns=classes,
        ),
        'jk_bar_year': year_result['jk_bar'],
        'jk_bar_type': type_result['jk_bar'],
        'years_list': years_list,
        'types_list': types_list,
        'year_to_i': year_to_i,
        'type_to_i': type_to_i,
        'max_N': max_N,
    }

    if cache_path:
        cache_payload = {
            key: value
            for key, value in results.items()
            if key not in {'year_to_i', 'type_to_i', 'max_N'}
        }
        joblib.dump(cache_payload, cache_path)
        print("Unified Jackknife results saved.")

    return results


def summarize_auc_grid(sample_metadata, fold_tensor, row_order, col_order,
                       row_metadata, col_metadata, row_name='row',
                       col_name='col', auc_divisor='max_N'):
    """Build a 2-D AUC summary table from jackknife fold tensors.

    This covers the repeated notebook pattern:
    metadata mask -> extract ``fold_tensor[:, mask, i, j]`` -> compute AUC mean
    and jackknife error -> append to a dataframe.

    Parameters
    ----------
    sample_metadata : pandas.DataFrame
        Metadata aligned to axis 1 of ``fold_tensor``.
    fold_tensor : numpy.ndarray
        Shape (max_N, n_samples, n_rows, n_cols).
    row_order, col_order : sequence
        Ordered category values corresponding to axes 2 and 3.
    row_metadata, col_metadata : str
        Metadata column names used to build the boolean mask.
    row_name, col_name : str, optional
        Output column names for the row and column categories.
    auc_divisor : str, optional
        'max_N' or 'max_N_minus_1'.

    Returns
    -------
    pandas.DataFrame
        Columns: row_name, col_name, 'auc', 'error', 'n_sub'.
    """
    records = []
    for row_i, row_val in enumerate(row_order):
        for col_j, col_val in enumerate(col_order):
            mask = (
                (sample_metadata[row_metadata] == row_val) &
                (sample_metadata[col_metadata] == col_val)
            ).to_numpy()
            summary = compute_auc_summary(
                extract_masked_fold_matrix(fold_tensor, mask, row_i, col_j),
                auc_divisor=auc_divisor,
            )
            records.append({
                row_name: row_val,
                col_name: col_val,
                'auc': summary['auc'],
                'error': summary['error'],
                'n_sub': summary['n'],
            })
    return pd.DataFrame(records)


def welch_ttest_from_summary(mean_A, se_A, n_A, mean_B, se_B, n_B,
                             alternative='two-sided'):
    """Welch-style test using summary stats from independent jackknife groups.

    Parameters
    ----------
    mean_A, mean_B : float
        Group means.
    se_A, se_B : float
        Group jackknife standard errors.
    n_A, n_B : int
        Number of folds/samples contributing to each group.
    alternative : str, optional
        'two-sided' (default), 'greater', or 'less'.

    Returns
    -------
    dict
        Keys: 'p_value', 'statistic', 'dof', 'se_diff'.
    """
    if n_A < 2 or n_B < 2 or (se_A == 0 and se_B == 0):
        return {'p_value': 1.0, 'statistic': 0.0, 'dof': np.nan, 'se_diff': 0.0}

    se_diff = float(np.sqrt(se_A ** 2 + se_B ** 2))
    if se_diff == 0:
        return {'p_value': 1.0, 'statistic': 0.0, 'dof': np.nan, 'se_diff': 0.0}

    t_stat = float((mean_A - mean_B) / se_diff)
    df_num = (se_A ** 2 + se_B ** 2) ** 2
    df_den = (se_A ** 4 / (n_A - 1)) + (se_B ** 4 / (n_B - 1))
    dof = float(df_num / df_den) if df_den != 0 else np.nan

    if alternative == 'greater':
        p_value = float(stats.t.sf(t_stat, dof))
    elif alternative == 'less':
        p_value = float(stats.t.cdf(t_stat, dof))
    else:
        p_value = float(2 * stats.t.sf(abs(t_stat), dof))

    return {
        'p_value': p_value,
        'statistic': t_stat,
        'dof': dof,
        'se_diff': se_diff,
    }


# ---------------------------------------------------------------------------
# Plotting utility functions
# ---------------------------------------------------------------------------

def _load_plot_module():
    """Load src/plot.py lazily to avoid notebook-side import boilerplate."""
    import importlib.util

    plot_path = os.path.join(os.path.dirname(__file__), 'plot.py')
    spec = importlib.util.spec_from_file_location('plot_utils', plot_path)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    return plot


def _resolve_plot_spec(spec, key, order_idx, default=None):
    """Resolve a per-curve style spec from a callable, dict, sequence, or scalar."""
    if spec is None:
        return default
    if callable(spec):
        return spec(key, order_idx)
    if isinstance(spec, dict):
        return spec.get(key, default)
    if isinstance(spec, str):
        return spec
    try:
        return spec[order_idx]
    except (TypeError, KeyError, IndexError):
        return default


def _resolve_group_index(axis_order, group, order_idx, group_index_fn=None):
    """Resolve the tensor index for a plotted group."""
    if group_index_fn is not None:
        return group_index_fn(group, order_idx)
    if isinstance(axis_order, dict):
        return axis_order[group]
    return axis_order.index(group)


def plot_group_topn_jackknife(
    jk_topn,
    sample_metadata,
    group_order,
    axis_order,
    metadata_col,
    baseline_group,
    color_dict,
    save_path=None,
    marker=None,
    figsize=(5.5, 3.5),
    alpha_fill=0.4,
    linewidth=1.5,
    markersize=4,
    legend_loc='lower right',
    legend_kwargs=None,
    x_anchor=1,
    x_anchor_index=0,
    bracket_step=0.45,
    bracket_hook_depth=0.35,
    bracket_fontsize=11,
    bracket_lw=1.0,
    bracket_clip_on=False,
    stats_fn=compute_masked_jackknife_topn_stats,
    group_values=None,
    mask_fn=None,
    group_index_fn=None,
    pvalue_fn=None,
    label_fn=None,
    ns=None,
    jitter_width=0.0,
    xlim=None,
    xticks=None,
    xlabel='N',
    ylabel='Top-N Accuracy',
    title=None,
    ylim=None,
    rotate_xticks=True,
    rotate_kwargs=None,
    auto_left_limit_from_brackets=False,
    auto_left_limit_pad=0.25,
    show=False,
):
    """Plot jackknife Top-N curves for a categorical grouping tensor slice.

    This covers the repeated notebook pattern:
    mask extraction -> per-group jackknife stats -> optional significance
    testing against a baseline -> curve plotting -> optional save.

    Parameters
    ----------
    jk_topn : numpy.ndarray
        Jackknife tensor with Top-N on axis 0 and sample on axis 1.
    sample_metadata : pandas.DataFrame
        Metadata aligned to axis 1 of ``jk_topn``.
    group_order : sequence
        Ordered groups to plot.
    axis_order : sequence or dict
        Ordered labels or explicit mapping for the grouping axis in ``jk_topn``.
    metadata_col : str
        Metadata column used to build group masks when ``mask_fn`` and
        ``group_values`` are not provided.
    baseline_group : hashable
        Group used as the significance baseline.
    color_dict : dict, sequence, callable, or scalar
        Per-group color specification.
    save_path : str, optional
        If provided, save the figure via ``save_fig``.
    marker : str, sequence, dict, or callable, optional
        Per-group marker specification.
    figsize : tuple, optional
        Matplotlib figure size.
    alpha_fill, linewidth, markersize, legend_loc, legend_kwargs : optional
        Passed to ``plot_topn_accuracy_curves``.
    x_anchor : float or None, optional
        Fixed x position where brackets attach. When ``None``, anchor on the
        baseline curve's point at ``x_anchor_index``.
    x_anchor_index : int, optional
        Curve point index used for bracket attachment.
    bracket_step, bracket_hook_depth, bracket_fontsize, bracket_lw,
    bracket_clip_on : optional
        Passed to ``draw_significance_brackets``.
    stats_fn : callable, optional
        Function called as ``stats_fn(jk_topn, mask, *group_indices)``.
    group_values : array-like, optional
        Precomputed group labels aligned to axis 1 of ``jk_topn``.
    mask_fn : callable, optional
        Custom mask builder called as ``mask_fn(group)``.
    group_index_fn : callable, optional
        Custom tensor index resolver called as ``group_index_fn(group, order_idx)``.
    pvalue_fn : callable, optional
        Called for each non-baseline group as ``pvalue_fn(**context)``.
        Defaults to ``jackknife_auc_unpaired_test`` on fold matrices.
    label_fn : callable, optional
        Called as ``label_fn(group, order_idx, stats)``. When omitted, the
        legend label defaults to the group name and AUC is appended by the
        plotting helper when available.
    ns : array-like, optional
        X values. Defaults to ``1..max_N``.
    jitter_width : float, optional
        Horizontal curve jitter width.
    xlim, xticks, xlabel, ylabel, title, ylim : optional
        Common axis formatting.
    rotate_xticks : bool, optional
        Whether to rotate x tick labels.
    rotate_kwargs : dict, optional
        Passed to ``rotate_xticklabels``.
    auto_left_limit_from_brackets : bool, optional
        If True, expand the left x-limit to make room for bracket stacks.
    auto_left_limit_pad : float, optional
        Extra padding used with ``auto_left_limit_from_brackets``.
    show : bool, optional
        Whether to call ``plt.show()`` before returning.

    Returns
    -------
    tuple
        ``(fig, ax, curves, bracket_info)``.
    """
    import matplotlib.pyplot as plt

    plot = _load_plot_module()
    plot.init_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    max_N = jk_topn.shape[0]
    if ns is None:
        ns = np.arange(1, max_N + 1)
    x_base = np.asarray(ns, dtype=float)
    n_curves = len(group_order)

    if mask_fn is None:
        if group_values is None:
            group_values = sample_metadata[metadata_col]
        group_values = np.asarray(group_values)

        def mask_fn(group):
            return group_values == group

    def get_x_positions(order_idx):
        if jitter_width > 0 and n_curves > 1:
            offset = (order_idx - (n_curves - 1) / 2) * jitter_width
            return x_base + offset
        return x_base

    records = []
    baseline_record = None

    for order_idx, group in enumerate(group_order):
        group_index = _resolve_group_index(axis_order, group, order_idx, group_index_fn)
        group_mask = np.asarray(mask_fn(group), dtype=bool)
        if isinstance(group_index, tuple):
            stats = stats_fn(jk_topn, group_mask, *group_index)
        else:
            stats = stats_fn(jk_topn, group_mask, group_index)

        record = {
            'group': group,
            'order_idx': order_idx,
            'group_index': group_index,
            'mask': group_mask,
            'stats': stats,
        }
        records.append(record)
        if group == baseline_group:
            baseline_record = record

    if baseline_record is None:
        raise ValueError(f"baseline_group {baseline_group!r} not found in group_order")

    baseline_fold_matrix = baseline_record['stats']['fold_matrix']
    baseline_y = baseline_record['stats']['means'][x_anchor_index]
    baseline_x = (
        get_x_positions(baseline_record['order_idx'])[x_anchor_index]
        if x_anchor is None else x_anchor
    )

    curves = []
    bracket_info = []

    for record in records:
        group = record['group']
        order_idx = record['order_idx']
        stats = record['stats']
        label = group if label_fn is None else label_fn(group, order_idx, stats)

        curves.append({
            'label': label,
            'means': stats['means'],
            'se': stats.get('se'),
            'auc': stats.get('auc'),
            'se_auc': stats.get('se_auc'),
            'color': _resolve_plot_spec(color_dict, group, order_idx),
            'marker': _resolve_plot_spec(marker, group, order_idx, 'o'),
        })

        if group == baseline_group:
            continue

        if pvalue_fn is None:
            p_val = jackknife_auc_unpaired_test(baseline_fold_matrix, stats['fold_matrix'])
        else:
            p_val = pvalue_fn(
                baseline_group=baseline_group,
                group=group,
                baseline_index=baseline_record['group_index'],
                group_index=record['group_index'],
                baseline_mask=baseline_record['mask'],
                group_mask=record['mask'],
                baseline_stats=baseline_record['stats'],
                group_stats=stats,
                baseline_fold_matrix=baseline_fold_matrix,
                fold_matrix=stats['fold_matrix'],
                jk_topn=jk_topn,
                sample_metadata=sample_metadata,
            )

        star = get_significance_stars(p_val)
        if star:
            bracket_info.append({
                'y_curr': stats['means'][x_anchor_index],
                'x_dot': get_x_positions(order_idx)[x_anchor_index],
                'star': star,
            })

    plot.plot_topn_accuracy_curves(
        curves,
        ns=ns,
        ax=ax,
        jitter_width=jitter_width,
        alpha_fill=alpha_fill,
        linewidth=linewidth,
        markersize=markersize,
        legend_loc=legend_loc,
        legend_kwargs=legend_kwargs,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    if bracket_info:
        plot.draw_significance_brackets(
            ax,
            bracket_info,
            y_base=baseline_y,
            x_anchor=baseline_x,
            step=bracket_step,
            hook_depth=bracket_hook_depth,
            fontsize=bracket_fontsize,
            lw=bracket_lw,
            clip_on=bracket_clip_on,
        )

    if auto_left_limit_from_brackets:
        left_limit = 0
        if bracket_info:
            left_limit = baseline_x - bracket_hook_depth - bracket_step * len(bracket_info) - auto_left_limit_pad
        right_limit = max_N + 0.5 if xlim is None else xlim[1]
        xlim = (left_limit, right_limit)

    if xlim is None:
        xlim = (0, max_N)
    if xticks is None:
        xticks = range(0, max_N + 1)

    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

    if rotate_xticks:
        rotate_kwargs = rotate_kwargs or {}
        plot.rotate_xticklabels(ax, **rotate_kwargs)

    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    if show:
        plt.show()

    return fig, ax, curves, bracket_info


def plot_precomputed_topn_jackknife(
    curve_data,
    baseline_label=None,
    fold_matrices=None,
    save_path=None,
    marker=None,
    figsize=(5.5, 3.5),
    alpha_fill=0.4,
    linewidth=1.5,
    markersize=4,
    legend_loc='lower right',
    legend_kwargs=None,
    pvalue_fn=None,
    label_fn=None,
    color_dict=None,
    ns=None,
    jitter_width=0.0,
    x_anchor=None,
    x_anchor_index=0,
    bracket_step=0.45,
    bracket_hook_depth=0.35,
    bracket_fontsize=11,
    bracket_lw=1.0,
    bracket_clip_on=False,
    xlim=None,
    xticks=None,
    xlabel='N',
    ylabel='Top-N Accuracy',
    title=None,
    ylim=None,
    rotate_xticks=True,
    rotate_kwargs=None,
    auto_left_limit_from_brackets=False,
    auto_left_limit_pad=0.25,
    show=False,
):
    """Plot Top-N curves from precomputed summaries with optional brackets.

    Parameters
    ----------
    curve_data : sequence of dict
        Precomputed per-curve summaries. Each item must contain ``label`` and
        ``means``; optional fields include ``se``, ``auc``, ``se_auc``, and
        any custom metadata consumed by ``label_fn``.
    baseline_label : hashable, optional
        Baseline label for significance testing and bracket anchoring.
    fold_matrices : dict, optional
        Mapping from label to fold matrix for significance testing.
    save_path : str, optional
        If provided, save the figure via ``save_fig``.
    marker : str, sequence, dict, or callable, optional
        Per-curve marker specification.
    figsize : tuple, optional
        Matplotlib figure size.
    alpha_fill, linewidth, markersize, legend_loc, legend_kwargs : optional
        Passed to the underlying plotting helper.
    pvalue_fn : callable, optional
        Called as ``pvalue_fn(**context)`` for each non-baseline curve.
    label_fn : callable, optional
        Called as ``label_fn(label, order_idx, curve_item)``.
    color_dict : dict, sequence, callable, or scalar, optional
        Per-curve color specification.
    ns, jitter_width, x_anchor, x_anchor_index, bracket_step,
    bracket_hook_depth, bracket_fontsize, bracket_lw, bracket_clip_on,
    xlim, xticks, xlabel, ylabel, title, ylim, rotate_xticks,
    rotate_kwargs, auto_left_limit_from_brackets, auto_left_limit_pad,
    show : optional
        Axis/bracket formatting options matching ``plot_group_topn_jackknife``.

    Returns
    -------
    tuple
        ``(fig, ax, curves, bracket_info)``.
    """
    import matplotlib.pyplot as plt

    plot = _load_plot_module()
    plot.init_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    labels = None
    if label_fn is not None:
        labels = lambda label, order_idx: label_fn(label, order_idx, curve_data[order_idx])

    ax, info = plot.plot_precomputed_topn_accuracy(
        curve_data=curve_data,
        ax=ax,
        ns=ns,
        baseline_label=baseline_label,
        fold_matrices=fold_matrices,
        pvalue_fn=pvalue_fn,
        labels=labels,
        colors=color_dict,
        markers=marker,
        jitter_width=jitter_width,
        alpha_fill=alpha_fill,
        linewidth=linewidth,
        markersize=markersize,
        legend_loc=legend_loc,
        legend_kwargs=legend_kwargs,
        x_anchor_index=x_anchor_index,
        bracket_step=bracket_step,
        bracket_hook_depth=bracket_hook_depth,
        bracket_fontsize=bracket_fontsize,
        bracket_lw=bracket_lw,
        bracket_clip_on=bracket_clip_on,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    curves = info['curves']
    bracket_info = info['bracket_info']
    baseline_x = info['baseline_x'] if x_anchor is None else x_anchor

    if bracket_info and x_anchor is not None:
        ax.lines.clear()
        plot.plot_precomputed_topn_accuracy(
            curve_data=curve_data,
            ax=ax,
            ns=ns,
            baseline_label=baseline_label,
            fold_matrices=fold_matrices,
            pvalue_fn=pvalue_fn,
            labels=labels,
            colors=color_dict,
            markers=marker,
            jitter_width=jitter_width,
            alpha_fill=alpha_fill,
            linewidth=linewidth,
            markersize=markersize,
            legend_loc=legend_loc,
            legend_kwargs=legend_kwargs,
            draw_brackets=False,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        plot.draw_significance_brackets(
            ax,
            bracket_info,
            y_base=info['baseline_y'],
            x_anchor=baseline_x,
            step=bracket_step,
            hook_depth=bracket_hook_depth,
            fontsize=bracket_fontsize,
            lw=bracket_lw,
            clip_on=bracket_clip_on,
        )

    if auto_left_limit_from_brackets:
        left_limit = 0
        if bracket_info:
            left_limit = baseline_x - bracket_hook_depth - bracket_step * len(bracket_info) - auto_left_limit_pad
        if xlim is None:
            right_limit = len(curve_data[0]['means']) + 0.5
        else:
            right_limit = xlim[1]
        xlim = (left_limit, right_limit)

    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

    if rotate_xticks:
        rotate_kwargs = rotate_kwargs or {}
        plot.rotate_xticklabels(ax, **rotate_kwargs)

    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)
    if show:
        plt.show()

    return fig, ax, curves, bracket_info


def plot_persistence_topn_jackknife(
    jk_topn,
    sample_metadata,
    exp_i,
    mat_i,
    time_labels,
    mat_name,
    save_path=None,
    title=None,
    exp_names=('exp201603', 'exp201709'),
    global_times=('Immediately', '5min', '1hr', '18hr', '24hr', '48hr', '72hr'),
    global_markers=('o', 's', '^', 'D', 's', '^', 'D'),
    time_to_sec=None,
    jitter_width=0.3,
    legend_loc='lower right',
    show=False,
):
    """Plot persistence Top-N jackknife curves from a precomputed tensor slice.

    Parameters
    ----------
    jk_topn : numpy.ndarray
        Jackknife tensor with shape ``(max_N, n_samples, n_exp, n_material, n_time)``.
    sample_metadata : pandas.DataFrame
        Metadata aligned to axis 1 of ``jk_topn``.
    exp_i, mat_i : int
        Tensor indices for experiment and material.
    time_labels : sequence of str
        Ordered persistence times to plot for the selected experiment.
    mat_name : str
        Material label used to filter ``sample_metadata``.
    save_path : str, optional
        If provided, save the figure via ``save_fig``.
    title : str, optional
        Figure title.
    exp_names, global_times, global_markers : sequence, optional
        Plot metadata shared across persistence figures.
    time_to_sec : dict, optional
        Mapping from persistence label to seconds. Defaults to the notebook mapping.
    jitter_width : float, optional
        Horizontal offset width for overlapping curves.
    legend_loc : str, optional
        Legend location.
    show : bool, optional
        Whether to call ``plt.show()`` before returning.

    Returns
    -------
    tuple
        ``(fig, ax, curves, bracket_info)``.
    """
    import seaborn as sns

    if time_to_sec is None:
        time_to_sec = {
            'Immediately': 1,
            '5min': 300,
            '1hr': 3600,
            '18hr': 64800,
            '24hr': 86400,
            '48hr': 172800,
            '72hr': 259200,
        }

    exp_name = exp_names[exp_i]
    global_colors = dict(zip(global_times, sns.color_palette('Blues_r', n_colors=11)[:7]))
    global_markers = dict(zip(global_times, global_markers))

    curve_data = []
    fold_matrices = {}

    for time_j, t_label in enumerate(time_labels):
        t_sec = time_to_sec[t_label]
        sample_mask = (
            (sample_metadata['experiment'] == exp_name) &
            (sample_metadata['surface_type'] == mat_name) &
            (sample_metadata['persistence_time_sec'] == t_sec)
        )
        valid_folds = np.where(sample_mask.to_numpy())[0]
        fold_matrix = jk_topn[:, valid_folds, exp_i, mat_i, time_j]

        if fold_matrix.size == 0 or np.all(np.isnan(fold_matrix)):
            continue

        stats = compute_jackknife_topn_stats(fold_matrix, auc_divisor='max_N_minus_1')
        curve_data.append({
            'label': t_label,
            'means': stats['means'],
            'se': stats['se'],
            'auc': stats['auc'],
            'se_auc': stats['se_auc'],
            'color': global_colors[t_label],
            'marker': global_markers[t_label],
        })
        fold_matrices[t_label] = fold_matrix

    max_N = jk_topn.shape[0]
    return plot_precomputed_topn_jackknife(
        curve_data,
        baseline_label=time_labels[0],
        fold_matrices=fold_matrices,
        save_path=save_path,
        marker=lambda label, order_idx: global_markers[label],
        color_dict=lambda label, order_idx: global_colors[label],
        pvalue_fn=lambda baseline_label, label, baseline_fold_matrix, fold_matrix, **kwargs: (
            jackknife_auc_unpaired_test(
                baseline_fold_matrix,
                fold_matrix,
                auc_divisor='max_N_minus_1',
            )
        ),
        jitter_width=jitter_width,
        alpha_fill=0.3,
        linewidth=1.5,
        markersize=5,
        legend_loc=legend_loc,
        legend_kwargs={'fontsize': 10, 'framealpha': 0.9},
        xlim=(0.5, max_N + 0.5),
        xticks=range(1, max_N + 1, 1 if max_N <= 20 else 2),
        title=title,
        show=show,
    )

def save_fig(fig, filepath, dpi=300, bbox_inches='tight'):
    """Save a matplotlib figure as both PDF and PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filepath : str
        Target path. If it ends with '.pdf' or '.png', the sibling file with
        the other extension is also written. If it has no extension, both
        '.pdf' and '.png' are appended.
    dpi : int, optional
        DPI for PNG output (default 300).
    bbox_inches : str, optional
        Bounding box setting (default 'tight').
    """
    root, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == '.pdf':
        pdf_path = filepath
        png_path = root + '.png'
    elif ext == '.png':
        pdf_path = root + '.pdf'
        png_path = filepath
    else:
        pdf_path = filepath + '.pdf'
        png_path = filepath + '.png'

    fig.savefig(pdf_path, bbox_inches=bbox_inches)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)


def add_manual_errorbars(ax, x, y, lower_errors=None, upper_errors=None,
                         color='black', capsize=3, elinewidth=1.0,
                         hide_lower_caps=False, zorder=5):
    """Add manual vertical error bars to an existing plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : array-like or scalar
        Bar/point coordinates.
    lower_errors, upper_errors : array-like or scalar, optional
        Asymmetric error sizes. When omitted, defaults to zeros.
    color : str, optional
        Error bar color (default 'black').
    capsize : float, optional
        Error bar cap size (default 3).
    elinewidth : float, optional
        Error bar line width (default 1.0).
    hide_lower_caps : bool, optional
        Hide the lower caps, useful for upper-only bar error bars.
    zorder : float, optional
        Matplotlib zorder (default 5).

    Returns
    -------
    tuple
        ``matplotlib.axes.Axes.errorbar`` return tuple.
    """
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    if y_arr.ndim != 1:
        raise ValueError('y must be a scalar or 1-D array-like.')

    n_points = y_arr.shape[0]

    if lower_errors is None:
        lower_arr = np.zeros(n_points, dtype=float)
    else:
        lower_arr = np.asarray(lower_errors, dtype=float)
        if lower_arr.ndim == 0:
            lower_arr = np.full(n_points, float(lower_arr))
        else:
            lower_arr = np.atleast_1d(lower_arr)
    if upper_errors is None:
        upper_arr = lower_arr.copy()
    else:
        upper_arr = np.asarray(upper_errors, dtype=float)
        if upper_arr.ndim == 0:
            upper_arr = np.full(n_points, float(upper_arr))
        else:
            upper_arr = np.atleast_1d(upper_arr)

    if x_arr.shape[0] != n_points:
        if x_arr.shape[0] == 1 and n_points > 1:
            x_arr = np.full(n_points, float(x_arr[0]))
        else:
            raise ValueError('x and y must have matching lengths.')

    if lower_arr.shape[0] != n_points or upper_arr.shape[0] != n_points:
        raise ValueError('Error arrays must match the length of y.')

    result = ax.errorbar(
        x_arr,
        y_arr,
        yerr=np.vstack([lower_arr, upper_arr]),
        fmt='none',
        c=color,
        capsize=capsize,
        elinewidth=elinewidth,
        zorder=zorder,
    )

    if hide_lower_caps:
        _, caplines, _ = result
        for cap_idx in range(0, len(caplines), 2):
            caplines[cap_idx].set_visible(False)

    return result


def annotate_pairwise_bar_significance(ax, coords_info, baseline_key,
                                       compare_keys=None, alternative='two-sided',
                                       line_y_start=None, line_step=0.021,
                                       hook_height=0.005, text_offset=-0.01,
                                       linewidth=1.0, color='black',
                                       fontsize=10):
    """Draw significance brackets from one baseline bar to comparison bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    coords_info : dict
        Mapping key -> {'x', 'y', 'err', 'n'} for each bar.
    baseline_key : hashable
        The baseline bar key in ``coords_info``.
    compare_keys : sequence, optional
        Which bars to compare to the baseline. Defaults to all others.
    alternative : str, optional
        'two-sided', 'greater', or 'less'.
    line_y_start : float, optional
        Initial y-position for the first bracket. Defaults to the tallest bar
        in the compared set plus a small offset.
    line_step : float, optional
        Vertical spacing between successive brackets.
    hook_height : float, optional
        Height of the vertical bracket hooks.
    text_offset : float, optional
        Offset applied to the significance text y-position.
    linewidth : float, optional
        Bracket line width.
    color : str, optional
        Bracket/text color.
    fontsize : float, optional
        Significance text size.

    Returns
    -------
    dict
        Keys: 'results' (list of comparison summaries),
        'global_max_y' (float), 'line_y_end' (float).
    """
    if baseline_key not in coords_info:
        return {'results': [], 'global_max_y': None, 'line_y_end': None}

    if compare_keys is None:
        compare_keys = [key for key in coords_info if key != baseline_key]
    compare_keys = [key for key in compare_keys if key in coords_info]

    relevant_keys = [baseline_key, *compare_keys]
    if line_y_start is None:
        line_y_start = max(
            coords_info[key]['y'] + coords_info[key].get('err', 0.0)
            for key in relevant_keys
        ) + 0.01

    current_line_y = line_y_start
    results = []
    baseline = coords_info[baseline_key]

    for key in compare_keys:
        other = coords_info[key]
        test = welch_ttest_from_summary(
            baseline['y'], baseline.get('err', 0.0), baseline.get('n', 0),
            other['y'], other.get('err', 0.0), other.get('n', 0),
            alternative=alternative,
        )
        star = get_significance_stars(test['p_value'])
        if not star:
            continue

        x1, x2 = baseline['x'], other['x']
        x_start, x_end = (x1, x2) if x1 < x2 else (x2, x1)
        ax.plot(
            [x_start, x_start, x_end, x_end],
            [current_line_y, current_line_y + hook_height,
             current_line_y + hook_height, current_line_y],
            lw=linewidth,
            c=color,
        )
        ax.text(
            (x_start + x_end) * 0.5,
            current_line_y + hook_height + text_offset,
            star,
            ha='center',
            va='bottom',
            color=color,
            fontsize=fontsize,
        )

        results.append({
            'key': key,
            'p_value': test['p_value'],
            'statistic': test['statistic'],
            'dof': test['dof'],
            'star': star,
            'line_y': current_line_y,
        })
        current_line_y += line_step

    return {
        'results': results,
        'global_max_y': current_line_y,
        'line_y_end': current_line_y,
    }


# ---------------------------------------------------------------------------
# Model utility functions
# ---------------------------------------------------------------------------

def load_or_fit_model(model, path, X, y):
    """Load cached model or fit and cache.

    Parameters
    ----------
    model : sklearn estimator
        Configured (but unfitted) model with .fit() method.
    path : str
        Path to cached model file (.pkl.bz2).
    X : array-like
        Training features.
    y : array-like
        Training labels.

    Returns
    -------
    sklearn estimator
        The best_estimator_ of the fitted GridSearchCV, or the fitted model
        itself if it does not have best_estimator_ attribute.
    """
    if os.path.exists(path):
        print(f"Loading pre-trained model from {path}...")
        try:
            model = joblib.load(path)
        except Exception as e:
            print(f"Warning: Failed to load cached model ({e}). Re-training...")
            model.fit(X, y)
            joblib.dump(model, path)
    else:
        print(f"Running model training... (this may take a while)")
        model.fit(X, y)
        joblib.dump(model, path)

    if hasattr(model, 'best_estimator_'):
        return model.best_estimator_
    return model


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_rf_grid_search(n_estimators=500, rf_random_state=1,
                        cv_random_state=11, n_jobs=14):
    """Create a GridSearchCV with RandomForestClassifier and RF_PARAM_GRID.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees (default 500).
    rf_random_state : int, optional
        Random state for RandomForestClassifier (default 1).
    cv_random_state : int, optional
        Random state for StratifiedKFold (default 11).
    n_jobs : int, optional
        Number of parallel jobs (default 14).

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Configured (unfitted) GridSearchCV object.
    """
    return GridSearchCV(
        RandomForestClassifier(n_estimators=n_estimators, random_state=rf_random_state),
        param_grid=RF_PARAM_GRID,
        cv=StratifiedKFold(3, shuffle=True, random_state=cv_random_state),
        n_jobs=n_jobs,
    )
