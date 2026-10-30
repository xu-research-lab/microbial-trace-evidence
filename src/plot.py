import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib import cm


def get_significance_stars(p, non_sig=''):
    """Return significance stars based on p-value.

    This is a local copy for use within plot.py. The canonical version
    lives in utils.py; both are kept in sync.

    Parameters
    ----------
    p : float
        p-value (non-negative).
    non_sig : str, optional
        Label for non-significant results (default '').

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


def plot_proba_heatmap(yhat, obs, ax=None, **kwargs):
    '''

    Parameters
    ----------
    yhat : df
        probability matrix. sample ID in row and predicted classes in column
    obs : tuple/list
        true classes for the samples. length is the same to yhat row count
    '''
    outcomes = yhat.columns.tolist()
    correct = 0
    if ax is None:
        fig, ax = plt.subplots()
    sns.heatmap(yhat, xticklabels=True, linewidths=1, ax=ax, **kwargs)
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    # count = len(yhat.index)
    for i, sid in enumerate(yhat.index):
        try:
            prob = np.nanargmax(yhat.loc[sid].values)
        except ValueError:
            #print(yhat.loc[sid])
            continue
        p = obs[i]
        pi = outcomes.index(p)
        #i = count - i - 1
        #print(sid, p, pi, i)
        kwd = dict(alpha=0.8, lw=1.5)
        if prob == pi:
            ax.add_patch(Rectangle((prob, i), 1, 1, fill=False, edgecolor='Lime', **kwd))
            correct += 1
        else:
            ax.add_patch(Rectangle((prob, i), 1, 1, fill=False, edgecolor='red', **kwd))
            ax.add_patch(Rectangle((pi, i), 1, 1, fill=False, edgecolor='DarkBlue', **kwd))
    ax.set_ylabel('samples')
    ax.set_xlabel('individuals')
    return correct, ax


def plot_rank_proba(ranks, n, order=None, colors=None, lines=None, markers=None, top_N=None, ax=None, **kwargs):
    '''
    Parameters
    ----------
    ranks : dict
        value is a list of ranks
    n : int
        the largest possible number
    '''
    if ax is None:
        fig, ax = plt.subplots()
    labels = []
    ticks = range(n + 1)
    if order is None:
        it = sorted(ranks)
    else:
        it = order
    for i, k in enumerate(it):
        if k not in ranks:
            continue
        data = ranks[k]
        if not data:
            # empty list
            continue
        values, base = np.histogram(data, normed=False, bins=ticks)
        values = values[1:]
        base = base[1:]
        #evaluate the cumulative
        cumulative = np.cumsum(values)
        line_type = '-' if lines is None else lines[i]
        color = None if colors is None else colors[i]
        marker = 'o' if markers is None else markers[i]
        ax.plot(base[:-1], cumulative / cumulative[-1], line_type+marker, color=color, **kwargs)

        if isinstance(k, tuple):
            l = ' '.join([str(i) for i in k])
        else:
            l = k
        if top_N is None:
            labels.append(l)
        else:
            accuracy = cumulative[top_N-1] / cumulative[-1]
            labels.append(l + ' (%.2f)' % accuracy)
    #if top_N is not None:
    #    ax.vlines(top_N, ymax=100, ymin=0, color='gray')
    ax.set_xlim(0, n)
    ax.set_xticks(ticks)
    ax.set_xlabel('N')
    ax.set_ylabel('Top-N Accuracy')
    ax.legend(labels=labels, loc='lower right')
    # ax.set_title('total %d unique subjects' % n)
    return ax

def plot_feature_imp(train_exp, model, norm=True, n=15, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    importance = model.feature_importances_
    if norm:
        importance = importance / importance.max()
    imp = pd.DataFrame({'importance': importance,
                        'taxonomy': train_exp.feature_metadata.taxonomy},
                       index=train_exp.feature_metadata.index)
    imp = imp.sort_values('importance', ascending=False)
    df = imp.iloc[:n]
    def find_highest(s):
        if isinstance(s, str):
            l = s.split(';')
            b = [len(i) > 3 for i in l]
            return np.array(l)[b][-1]
        else:
            return [None] * 7
    df = df.assign(taxa=df['taxonomy'].apply(find_highest))
    df.plot.barh(x='taxa', y='importance', ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('feature importance')
    ax.figure.tight_layout()
    return imp, ax


# ---------------------------------------------------------------------------
# Style utility functions
# ---------------------------------------------------------------------------

def init_plot_style(style='ticks'):
    """Reset matplotlib defaults and apply seaborn style.

    Parameters
    ----------
    style : str, optional
        Seaborn style name (default 'ticks').
    """
    plt.rcdefaults()
    sns.set(style=style)


def rotate_xticklabels(ax, rotation=45, ha='right', rotation_mode='anchor'):
    """Rotate x-axis tick labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    rotation : float, optional
        Rotation angle in degrees (default 45).
    ha : str, optional
        Horizontal alignment (default 'right').
    rotation_mode : str, optional
        Rotation mode (default 'anchor').
    """
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha,
             rotation_mode=rotation_mode)


# ---------------------------------------------------------------------------
# Top-N accuracy curve plotting
# ---------------------------------------------------------------------------

def plot_topn_accuracy_curves(curves, ns=None, ax=None, figsize=(5.5, 3.5),
                              jitter_width=0.0, alpha_fill=0.4,
                              linewidth=1.5, markersize=4,
                              markeredgewidth=0, show_error_bands=True,
                              ci_multiplier=1.96, xlabel='N',
                              ylabel='Top-N Accuracy', legend_loc='lower right',
                              legend_kwargs=None):
    """Plot Top-N accuracy curves with optional error bands and X-axis jitter.

    Parameters
    ----------
    curves : list of dict
        Each dict must have keys: 'means' (array), 'se' (array), 'label' (str),
        'color' (str), 'marker' (str, e.g. 'o').
        Optional keys: 'auc' (float), 'se_auc' (float), 'stars' (str).
    ns : array-like, optional
        X-axis values. Defaults to 0..len(means)-1.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size when creating new figure (default (5.5, 3.5)).
    jitter_width : float, optional
        Width of X-axis jitter for overlapping curves (default 0.0).
    alpha_fill : float, optional
        Transparency of error bands (default 0.4).
    linewidth : float, optional
        Line width (default 1.5).
    markersize : float, optional
        Marker size (default 4).
    markeredgewidth : float, optional
        Marker edge width (default 0).
    show_error_bands : bool, optional
        Whether to draw error bands (default True).
    ci_multiplier : float, optional
        Confidence interval multiplier (default 1.96 for 95% CI).
    xlabel : str, optional
        X-axis label (default 'N').
    ylabel : str, optional
        Y-axis label (default 'Top-N Accuracy').
    legend_loc : str, optional
        Legend location (default 'lower right').
    legend_kwargs : dict, optional
        Additional keyword arguments for legend.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_curves = len(curves)
    for i, curve in enumerate(curves):
        means = curve['means']
        se = curve.get('se')
        color = curve['color']
        marker = curve.get('marker', 'o')
        label = curve.get('label', '')

        # Compute x positions with jitter
        if ns is None:
            x = np.arange(len(means))
        else:
            x = np.asarray(ns, dtype=float)
        if jitter_width > 0 and n_curves > 1:
            offset = (i - (n_curves - 1) / 2) * jitter_width
            x_plot = x + offset
        else:
            x_plot = x

        # Build label with AUC if available
        if (
            'auc' in curve and 'se_auc' in curve and
            curve['auc'] is not None and curve['se_auc'] is not None
        ):
            auc_str = f"{curve['auc']:.3f}\u00B1{ci_multiplier * curve['se_auc']:.3f}"
            stars = curve.get('stars', '')
            if label:
                label = f"{label} ({auc_str}){stars}"
            else:
                label = f"({auc_str}){stars}"

        ax.plot(x_plot, means, '-' + marker, color=color, label=label,
                linewidth=linewidth, markersize=markersize,
                markeredgewidth=markeredgewidth)

        if show_error_bands and se is not None:
            lower = np.clip(means - ci_multiplier * se, 0, 1)
            upper = np.clip(means + ci_multiplier * se, 0, 1)
            ax.fill_between(x_plot, lower, upper, color=color,
                            alpha=alpha_fill, edgecolor='none')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_kwargs is None:
        legend_kwargs = {}
    ax.legend(loc=legend_loc, **legend_kwargs)
    return ax


# ---------------------------------------------------------------------------
# Significance brackets
# ---------------------------------------------------------------------------

def draw_significance_brackets(ax, bracket_info, y_base, x_anchor=1,
                               step=0.45, hook_depth=0.35, fontsize=11,
                               lw=1.0, clip_on=False):
    """Draw left-side significance brackets comparing curves to a baseline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    bracket_info : list of dict
        Each dict must have keys: 'y_curr' (float), 'star' (str).
        Optional key: 'x_dot' (float) for jittered x position.
    y_base : float
        Y-coordinate of the baseline curve at the bracket attachment point.
    x_anchor : float, optional
        Starting x position for bracket attachment (default 1).
    step : float, optional
        Horizontal spacing between stacked brackets (default 0.45).
    hook_depth : float, optional
        How far left the bracket extends (default 0.35).
    fontsize : float, optional
        Font size for significance stars (default 11).
    lw : float, optional
        Line width for bracket lines (default 1.0).
    clip_on : bool, optional
        Whether to clip brackets to axes boundary (default False).
    """
    bracket_info = sorted(bracket_info, key=lambda b: abs(b['y_curr'] - y_base))
    x_backbone = x_anchor - hook_depth
    for b in bracket_info:
        y_curr = b['y_curr']
        star = b['star']
        x_dot = b.get('x_dot', x_anchor)
        ax.plot([x_dot, x_backbone, x_backbone, x_anchor],
                [y_curr, y_curr, y_base, y_base],
                lw=lw, c='black', clip_on=clip_on)
        ax.text(
            x_backbone - 0.05,
            (y_curr + y_base) / 2,
            star,
            ha='center',
            va='center',
            rotation=90,
            rotation_mode='anchor',
            color='black',
            fontsize=fontsize,
        )
        x_backbone -= step


def _resolve_group_spec(spec, group, order_idx, default=None):
    """Resolve a per-group style/index spec from a callable, dict, or sequence."""
    if spec is None:
        return default
    if callable(spec):
        return spec(group, order_idx)
    if isinstance(spec, dict):
        return spec.get(group, default)
    if isinstance(spec, str):
        return spec
    try:
        return spec[order_idx]
    except (TypeError, KeyError, IndexError):
        return default


def plot_grouped_topn_accuracy(
    fold_tensor,
    group_order,
    baseline_group,
    stats_fn,
    ax=None,
    figsize=(5.5, 3.5),
    sample_metadata=None,
    metadata_col=None,
    group_values=None,
    mask_fn=None,
    group_indices=None,
    pvalue_fn=None,
    star_fn=get_significance_stars,
    colors=None,
    labels=None,
    markers=None,
    ns=None,
    jitter_width=0.0,
    alpha_fill=0.4,
    linewidth=1.5,
    markersize=4,
    markeredgewidth=0,
    show_error_bands=True,
    ci_multiplier=1.96,
    legend_loc='lower right',
    legend_kwargs=None,
    draw_brackets=True,
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
    rotate_xlabels=True,
    rotate_kwargs=None,
):
    """Plot grouped Top-N accuracy curves and optional baseline significance.

    This wraps the repeated notebook pattern:
    group mask -> jackknife stats -> optional baseline p-value -> curve plotting
    -> optional significance brackets -> common axis formatting.

    Parameters
    ----------
    fold_tensor : numpy.ndarray
        Jackknife tensor whose first axis is Top-N and second axis is sample.
    group_order : sequence
        Ordered groups to plot.
    baseline_group : hashable
        Group used as the significance baseline.
    stats_fn : callable
        Function called as ``stats_fn(fold_tensor, mask, *group_idx)`` and
        expected to return a dict with at least ``means`` and ``se``. If
        significance testing is used, it should also expose ``fold_matrix``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple, optional
        Figure size when creating a new axes.
    sample_metadata : pandas.DataFrame, optional
        Metadata table aligned to axis 1 of ``fold_tensor``. Passed through to
        ``pvalue_fn`` and used with ``metadata_col`` when building masks.
    metadata_col : str, optional
        Metadata column to compare against each group when ``mask_fn`` is not
        provided.
    group_values : array-like, optional
        Precomputed values aligned to axis 1 of ``fold_tensor``. Useful when
        the mask should be based on a transformed metadata series.
    mask_fn : callable, optional
        Custom mask builder called as ``mask_fn(group)``.
    group_indices : callable, dict, sequence, or scalar, optional
        Selector for grouping-axis indices passed to ``stats_fn``. A callable
        receives ``(group, order_idx)`` and may return an int or a tuple.
    pvalue_fn : callable, optional
        Called for each non-baseline group as ``pvalue_fn(**context)`` where
        context includes baseline/current stats, masks, group names, indices,
        ``fold_tensor``, and ``sample_metadata``.
    star_fn : callable, optional
        Maps a p-value to a significance label.
    colors, labels, markers : callable, dict, sequence, or scalar, optional
        Per-group styling specs.
    ns : array-like, optional
        X-axis values. Defaults to ``1..max_N``.
    jitter_width : float, optional
        Horizontal offset width shared with ``plot_topn_accuracy_curves``.
    alpha_fill, linewidth, markersize, markeredgewidth, show_error_bands,
    ci_multiplier, legend_loc, legend_kwargs : optional
        Passed through to ``plot_topn_accuracy_curves``.
    draw_brackets : bool, optional
        Whether to annotate baseline-vs-group significance brackets.
    x_anchor_index : int, optional
        Curve point index used to anchor significance brackets.
    bracket_step, bracket_hook_depth, bracket_fontsize, bracket_lw,
    bracket_clip_on : optional
        Passed through to ``draw_significance_brackets``.
    xlim, xticks, xlabel, ylabel : optional
        Common axis formatting.
    rotate_xlabels : bool, optional
        Whether to call ``rotate_xticklabels`` on the output axes.
    rotate_kwargs : dict, optional
        Extra keyword arguments for ``rotate_xticklabels``.

    Returns
    -------
    tuple
        ``(ax, info)`` where ``info`` contains curves, bracket_info,
        baseline_x, baseline_y, and per-group stats.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if mask_fn is None:
        if group_values is None:
            if sample_metadata is None or metadata_col is None:
                raise ValueError(
                    "Provide mask_fn, or provide sample_metadata plus metadata_col, "
                    "or provide group_values."
                )
            group_values = sample_metadata[metadata_col]

        group_values = np.asarray(group_values)

        def mask_fn(group):
            return group_values == group

    if ns is None:
        ns = np.arange(1, fold_tensor.shape[0] + 1)
    x_base = np.asarray(ns, dtype=float)
    n_curves = len(group_order)

    def get_x_positions(order_idx):
        if jitter_width > 0 and n_curves > 1:
            offset = (order_idx - (n_curves - 1) / 2) * jitter_width
            return x_base + offset
        return x_base

    group_records = []
    baseline_record = None

    for order_idx, group in enumerate(group_order):
        group_idx = _resolve_group_spec(group_indices, group, order_idx, order_idx)
        mask = np.asarray(mask_fn(group), dtype=bool)
        if isinstance(group_idx, tuple):
            stats = stats_fn(fold_tensor, mask, *group_idx)
        else:
            stats = stats_fn(fold_tensor, mask, group_idx)

        record = {
            'group': group,
            'order_idx': order_idx,
            'group_idx': group_idx,
            'mask': mask,
            'stats': stats,
        }
        group_records.append(record)
        if group == baseline_group:
            baseline_record = record

    if baseline_record is None:
        raise ValueError(f"baseline_group {baseline_group!r} not found in group_order")

    curves = []
    bracket_info = []
    baseline_y = baseline_record['stats']['means'][x_anchor_index]
    baseline_x = get_x_positions(baseline_record['order_idx'])[x_anchor_index]

    for record in group_records:
        group = record['group']
        order_idx = record['order_idx']
        stats = record['stats']

        curves.append({
            'label': _resolve_group_spec(labels, group, order_idx, group),
            'means': stats['means'],
            'se': stats['se'],
            'auc': stats.get('auc'),
            'se_auc': stats.get('se_auc'),
            'color': _resolve_group_spec(colors, group, order_idx),
            'marker': _resolve_group_spec(markers, group, order_idx, 'o'),
        })

        if (
            draw_brackets and
            pvalue_fn is not None and
            group != baseline_group
        ):
            p_val = pvalue_fn(
                baseline_stats=baseline_record['stats'],
                group_stats=stats,
                baseline_group=baseline_group,
                group=group,
                baseline_index=baseline_record['group_idx'],
                group_index=record['group_idx'],
                baseline_mask=baseline_record['mask'],
                group_mask=record['mask'],
                fold_tensor=fold_tensor,
                sample_metadata=sample_metadata,
            )
            star = star_fn(p_val)
            if star:
                bracket_info.append({
                    'y_curr': stats['means'][x_anchor_index],
                    'x_dot': get_x_positions(order_idx)[x_anchor_index],
                    'star': star,
                })

    plot_topn_accuracy_curves(
        curves,
        ns=ns,
        ax=ax,
        jitter_width=jitter_width,
        alpha_fill=alpha_fill,
        linewidth=linewidth,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        show_error_bands=show_error_bands,
        ci_multiplier=ci_multiplier,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_loc=legend_loc,
        legend_kwargs=legend_kwargs,
    )

    if draw_brackets and bracket_info:
        draw_significance_brackets(
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

    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if rotate_xlabels:
        rotate_kwargs = rotate_kwargs or {}
        rotate_xticklabels(ax, **rotate_kwargs)

    info = {
        'curves': curves,
        'bracket_info': bracket_info,
        'baseline_x': baseline_x,
        'baseline_y': baseline_y,
        'group_records': group_records,
    }
    return ax, info


def plot_precomputed_topn_accuracy(
    curve_data,
    ax=None,
    figsize=(5.5, 3.5),
    ns=None,
    baseline_label=None,
    fold_matrices=None,
    pvalue_fn=None,
    star_fn=get_significance_stars,
    labels=None,
    colors=None,
    markers=None,
    jitter_width=0.0,
    alpha_fill=0.4,
    linewidth=1.5,
    markersize=4,
    markeredgewidth=0,
    show_error_bands=True,
    ci_multiplier=1.96,
    legend_loc='lower right',
    legend_kwargs=None,
    draw_brackets=True,
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
    rotate_xlabels=True,
    rotate_kwargs=None,
):
    """Plot Top-N curves from precomputed per-group summaries.

    Parameters
    ----------
    curve_data : sequence of dict
        Each item should include at least ``label`` and ``means``. Optional
        keys include ``se``, ``auc``, ``se_auc``, and any extra metadata used
        by ``pvalue_fn``.
    baseline_label : hashable, optional
        Label used as the significance baseline.
    fold_matrices : dict, optional
        Mapping from label to fold matrix for significance testing.
    pvalue_fn : callable, optional
        Called for each non-baseline label as ``pvalue_fn(**context)`` where
        context includes baseline/current curves, labels, and fold matrices.
    labels, colors, markers : callable, dict, sequence, or scalar, optional
        Per-curve styling specs.
    Other parameters
        Mirror ``plot_grouped_topn_accuracy`` and
        ``plot_topn_accuracy_curves``.

    Returns
    -------
    tuple
        ``(ax, info)`` where ``info`` contains curves, bracket_info,
        baseline_x, baseline_y, and the input curve records.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    curve_data = list(curve_data)
    if not curve_data:
        raise ValueError("curve_data must not be empty")

    if ns is None:
        ns = np.arange(1, len(curve_data[0]['means']) + 1)
    x_base = np.asarray(ns, dtype=float)
    n_curves = len(curve_data)

    def get_x_positions(order_idx):
        if jitter_width > 0 and n_curves > 1:
            offset = (order_idx - (n_curves - 1) / 2) * jitter_width
            return x_base + offset
        return x_base

    curve_records = []
    baseline_record = None

    for order_idx, data in enumerate(curve_data):
        label = data['label']
        record = {
            'label': label,
            'order_idx': order_idx,
            'data': data,
            'fold_matrix': None if fold_matrices is None else fold_matrices.get(label),
        }
        curve_records.append(record)
        if label == baseline_label:
            baseline_record = record

    curves = []
    bracket_info = []
    baseline_x = None
    baseline_y = None

    if baseline_label is not None:
        if baseline_record is None:
            raise ValueError(f"baseline_label {baseline_label!r} not found in curve_data")
        baseline_x = get_x_positions(baseline_record['order_idx'])[x_anchor_index]
        baseline_y = baseline_record['data']['means'][x_anchor_index]

    for record in curve_records:
        data = record['data']
        label = record['label']
        order_idx = record['order_idx']
        resolved_color = _resolve_group_spec(colors, label, order_idx)
        resolved_marker = _resolve_group_spec(markers, label, order_idx)
        curve = {
            'label': _resolve_group_spec(labels, label, order_idx, label),
            'means': data['means'],
            'se': data.get('se'),
            'auc': data.get('auc'),
            'se_auc': data.get('se_auc'),
            'color': data.get('color', resolved_color) if resolved_color is None else resolved_color,
            'marker': data.get('marker', 'o') if resolved_marker is None else resolved_marker,
            'stars': data.get('stars', ''),
        }
        curves.append(curve)

        if (
            draw_brackets and
            baseline_record is not None and
            pvalue_fn is not None and
            label != baseline_label and
            record['fold_matrix'] is not None and
            baseline_record['fold_matrix'] is not None
        ):
            p_val = pvalue_fn(
                baseline_curve=baseline_record['data'],
                curve=data,
                baseline_label=baseline_label,
                label=label,
                baseline_fold_matrix=baseline_record['fold_matrix'],
                fold_matrix=record['fold_matrix'],
            )
            star = star_fn(p_val)
            if star:
                bracket_info.append({
                    'y_curr': data['means'][x_anchor_index],
                    'x_dot': get_x_positions(order_idx)[x_anchor_index],
                    'star': star,
                })

    plot_topn_accuracy_curves(
        curves,
        ns=ns,
        ax=ax,
        jitter_width=jitter_width,
        alpha_fill=alpha_fill,
        linewidth=linewidth,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        show_error_bands=show_error_bands,
        ci_multiplier=ci_multiplier,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_loc=legend_loc,
        legend_kwargs=legend_kwargs,
    )

    if draw_brackets and bracket_info and baseline_y is not None:
        draw_significance_brackets(
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

    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if rotate_xlabels:
        rotate_kwargs = rotate_kwargs or {}
        rotate_xticklabels(ax, **rotate_kwargs)

    info = {
        'curves': curves,
        'bracket_info': bracket_info,
        'baseline_x': baseline_x,
        'baseline_y': baseline_y,
        'curve_records': curve_records,
    }
    return ax, info


# ---------------------------------------------------------------------------
# Boxplot + stripplot overlay
# ---------------------------------------------------------------------------

def plot_boxplot_with_strip(data, x, y, hue, order=None, hue_order=None,
                            palette=None, ax=None, figsize=(5, 3.5),
                            fliersize=0, strip_size=2, strip_alpha=0.5,
                            strip_jitter=True, p_values=None,
                            sig_y_offset=0.03, non_sig='n.s.',
                            ylabel=None, xlabel=None,
                            legend_ncol=2, legend_loc='upper right',
                            legend_kwargs=None, xtick_rotation=45):
    """Overlay stripplot on boxplot with optional significance markers.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    x, y, hue : str
        Column names for boxplot/stripplot.
    order : list, optional
        Order for x categories.
    hue_order : list, optional
        Order for hue categories.
    palette : str or dict, optional
        Color palette.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size (default (5, 3.5)).
    fliersize : float, optional
        Size of outlier markers (default 0).
    strip_size : float, optional
        Size of strip points (default 2).
    strip_alpha : float, optional
        Transparency of strip points (default 0.5).
    strip_jitter : bool, optional
        Whether to jitter strip points (default True).
    p_values : dict, optional
        Mapping from x-category to p-value for significance markers.
    sig_y_offset : float, optional
        Y offset for significance text below the plot (default 0.03).
    non_sig : str, optional
        Label for non-significant results (default 'n.s.').
    ylabel, xlabel : str, optional
        Axis labels.
    legend_ncol : int, optional
        Number of legend columns (default 2).
    legend_loc : str, optional
        Legend location (default 'upper right').
    legend_kwargs : dict, optional
        Additional legend keyword arguments.
    xtick_rotation : float, optional
        X-tick label rotation (default 45).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(x=x, y=y, hue=hue, data=data, order=order,
                hue_order=hue_order, palette=palette,
                ax=ax, fliersize=fliersize)
    sns.stripplot(x=x, y=y, hue=hue, data=data, order=order,
                  hue_order=hue_order, palette=palette,
                  ax=ax, dodge=True, size=strip_size,
                  jitter=strip_jitter, alpha=strip_alpha)

    # Significance markers
    if p_values is not None:
        y_min = data[y].min()
        for i, cat in enumerate(order or data[x].unique()):
            p_val = p_values.get(cat)
            if p_val is not None:
                sig = get_significance_stars(p_val, non_sig=non_sig)
                ax.text(i, y_min - sig_y_offset, sig, ha='center',
                        va='top', color='black', fontsize=10)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    n_cats = len(hue_order) if hue_order else data[hue].nunique()
    if legend_kwargs is None:
        legend_kwargs = {}
    ax.legend(handles[:n_cats], labels[:n_cats], loc=legend_loc,
              ncol=legend_ncol, **legend_kwargs)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if xtick_rotation:
        rotate_xticklabels(ax, rotation=xtick_rotation)
    return ax


# ---------------------------------------------------------------------------
# Simulation heatmap
# ---------------------------------------------------------------------------

def plot_simulation_heatmap(data, title='TOP-N Accuracy', ax=None,
                            figsize=(9, 9), cmap="Blues", vmin=0.3, vmax=1,
                            xlabel='Samples per Subject',
                            ylabel='Number of Individuals', fontsize=12):
    """Plot simulation heatmap of accuracy vs. individual/sample counts.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Pivot table: rows=individuals, cols=samples_per_subject.
    title : str, optional
        Plot title (default 'TOP-N Accuracy').
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size (default (9, 9)).
    cmap : str, optional
        Colormap (default 'Blues').
    vmin, vmax : float, optional
        Color scale limits (default 0.3, 1).
    xlabel, ylabel : str, optional
        Axis labels.
    fontsize : float, optional
        Font size for labels and title (default 12).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data, cmap=cmap, linecolor='lightgray', square=True,
                vmin=vmin, vmax=vmax, ax=ax)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title, fontsize=fontsize)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_title('Accuracy', fontsize=fontsize)
    return ax
