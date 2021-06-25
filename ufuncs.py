"""
unfinished.

some handy functions that I have written. Extremely helpful when dealing with stock data,
which could have missing values, and the funcs deal with them properly. Not all of them are used in the pj.
"""


import numpy as np
import pandas as pd


def rolling_mean(arr, window, min_periods=1, axis=0):
    """cal rolling mean for array of any shape """
    ret = np.nancumsum(arr, axis=axis)
    slc1 = [slice(None)] * len(arr.shape)
    slc1[axis] = slice(window, arr.shape[axis])
    slc2 = [slice(None)] * len(arr.shape)
    slc2[axis] = slice(0, arr.shape[axis] - window)
    slc3 = [slice(None)] * len(arr.shape)
    slc3[axis] = slice(0, window - 1)

    ret[tuple(slc1)] = ret[tuple(slc1)] - ret[tuple(slc2)]
    ret[np.isnan(arr)] = np.nan

    # if we simply divide by window, may not be appropriate
    valid_samples_count = np.cumsum(~np.isnan(arr), axis=axis)
    valid_samples_count[tuple(slc1)] = valid_samples_count[tuple(slc1)] - valid_samples_count[tuple(slc2)]
    ret[valid_samples_count < min_periods] = np.nan

    return ret / valid_samples_count


def ffill(arr, axis, inplace=False):
    """对任意维数组沿某轴前值填充. 目前速度不如转DataFrame进行ffill. 不过高维数组只能用这个. """
    mask = np.isnan(arr)
    target_shape = [1, ] * len(arr.shape)
    target_shape[axis] = -1
    idx = np.where(~mask, np.arange(mask.shape[axis]).reshape(target_shape),
                   0)  # 将np.arange变形为除了axis外维数都为1的向量, 否则无法broadcast
    np.maximum.accumulate(idx, axis=axis, out=idx)  # 对idx做累计最大. 由于idx本身是递增的，只有被替换为0的那些部分会以前值填充
    slc = []
    for i in range(len(arr.shape)):
        if i == axis:
            slc.append(idx[mask])
        else:
            slc.append(np.nonzero(mask)[i])
    slc = tuple(slc)  # 构造一个除了目标维度的index被替换为其前值的index外, 其他维度index不变的slicing
    if inplace == True:
        arr[mask] = arr[slc]
    else:
        out = arr.copy()
        out[mask] = out[slc]
        return out


def ts_rank(arr, window):
    """输入2维向量, 输出其time series rank."""

    def rank(arr):
        return (arr <= arr[-1]).sum()  # 输入一列, 返回其最后一个元素的rank

    return pd.DataFrame(arr).rolling(window=window, axis=0, min_periods=1).apply(func=rank, raw=True).to_numpy()


def rolling_beta(arr_X, arr_Y, window):
    """computes rolling beta."""
    rolling_var = pd.DataFrame(arr_X).rolling(window=window, axis=0, min_periods=1).var()
    arr_X = np.repeat(np.expand_dims(arr_X, 1), arr_Y.shape[1], axis=1)
    rolling_cov = pd.DataFrame(arr_X).rolling(window=window, axis=0, min_periods=1).cov(pd.DataFrame(arr_Y))
    return rolling_cov.to_numpy() / rolling_var.to_numpy()


def ts_delay(arr, window=1, axis=0):
    """delay by window"""
    ret = arr.copy()
    if window >= 0:
        slc1 = [slice(None)] * len(arr.shape)
        slc1[axis] = slice(window, arr.shape[axis])
        slc2 = [slice(None)] * len(arr.shape)
        slc2[axis] = slice(0, arr.shape[axis] - window)
        slc3 = [slice(None)] * len(arr.shape)
        slc3[axis] = slice(0, window)
        ret[tuple(slc1)] = ret[tuple(slc2)]
        ret[tuple(slc3)] = np.nan
    else: # delay by negative, fetching future data
        slc1 = [slice(None)] * len(arr.shape)
        slc1[axis] = slice(-window, arr.shape[axis])
        slc2 = [slice(None)] * len(arr.shape)
        slc2[axis] = slice(0, window)
        slc3 = [slice(None)] * len(arr.shape)
        slc3[axis] = slice(window, arr.shape[axis])
        ret[tuple(slc2)] = ret[tuple(slc1)]
        ret[tuple(slc3)] = np.nan
    return ret


def ts_delta(arr, window, axis=0):
    """delta"""
    return arr - ts_delay(arr, window, axis)


def bollinger_band(arr, window, k1, k2=None, min_periods=5):
    """gives ts signal according to a bollinger band. Note: only for 2 dim array."""
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, 1)
    ret = np.full_like(arr, fill_value=np.nan)
    arr_rolling = pd.DataFrame(arr).rolling(min_periods=min(window, min_periods), axis=0, window=window)
    mean = arr_rolling.mean()
    std = arr_rolling.std()
    buy_mask = arr > (np.array(mean) + k1[0] * np.array(std))
    sell_mask = arr < (np.array(mean) - k1[1] * np.array(std))
    ret[buy_mask] = 1
    ret[sell_mask] = -1
    if k2:
        buy_mask = arr > (np.array(mean) + k2[0] * np.array(std))
        sell_mask = arr < (np.array(mean) - k2[1] * np.array(std))
        ret[buy_mask] = -1
        ret[sell_mask] = 1
    ret = pd.DataFrame(ret).fillna(method='ffill', axis=0,
                                   inplace=False).to_numpy()  # if no signal, retain previous signal
    ret[np.isnan(arr)] = np.nan
    return ret


def bollinger_filter(arr, axis, k1, k2=None):
    """bollinger band along an axis"""
    ret = np.full_like(arr, fill_value=np.nan)
    mean = arr.mean(axis, keepdims=True)
    std = arr.std(axis, keepdims=True)
    buy_mask = arr > (mean + k1[0] * std)
    sell_mask = arr < (mean - k1[1] * std)
    ret[buy_mask] = 1
    ret[sell_mask] = -1
    ret[(~buy_mask) & (~sell_mask) & (~np.isnan(arr))] = 0
    return ret


def ewma(arr, axis=0, alpha=None, halflife=None):
    """calculates ewma. receives either alpha or halflife.
    if nan is encountered, all previous weights are abandoned, unlike pandas behavior.
    sharpe lower than pandas."""
    if alpha == None:
        alpha = 1 - np.exp(-np.log(2) / halflife)
    n = arr.shape[axis]
    out = np.full(arr.shape, np.nan)
    for i in range(n):
        slc = [slice(None)] * len(arr.shape)
        slc[axis] = (i,)
        slc_prev = [slice(None)] * len(arr.shape)
        slc_prev[axis] = (i - 1,)
        if i == 0:
            prev = arr[tuple(slc)]
            out[tuple(slc)] = prev
        else:
            prev = arr[tuple(slc)] * alpha + np.nan_to_num(prev * (1 - alpha))
            out[tuple(slc)] = prev
    return out


def rolling_max_argmax(arr, window, axis=0):
    """returns rolling max and argmax arr, along given axis."""
    max_arr = np.full_like(arr, np.nan)
    argmax_arr = np.full_like(arr, np.nan)
    arr_filled = np.nan_to_num(arr, nan=-np.inf)
    for t in range(arr.shape[axis]):
        mask = [slice(None)] * len(arr.shape)
        mask[axis] = slice(max(t - window + 1, 0), t + 1)
        mask_not_all_nan = (~np.isnan(arr[tuple(mask)])).sum(axis) >= 1  # all nan columns need to be excluded
        mask_arr = [slice(None)] * len(arr.shape)
        mask_arr[axis] = t
        max_arr[tuple(mask_arr)][mask_not_all_nan] = np.nanmax(arr_filled[tuple(mask)], axis=axis)[mask_not_all_nan]
        argmax_arr[tuple(mask_arr)][mask_not_all_nan] = np.nanargmax(arr_filled[tuple(mask)], axis=axis)[
            mask_not_all_nan]
    return max_arr, argmax_arr


def k_slice(arr, axis, k, top=True):
    """返回沿某轴最大/小的k个数. 注意，该函数不处理nan，因此只能对日内维度做. """
    slc = [slice(None)] * len(arr.shape)
    if top == True:
        k = -k
        slc[axis] = slice(k, arr.shape[axis])
    else:
        slc[axis] = slice(0, k)

    topk_arr = np.take_along_axis(arr, np.argpartition(arr, axis=axis, kth=k), axis=axis)[tuple(slc)]
    return topk_arr


def top_cap(arr2d, pct, axis=1, nan=False):
    """top x% are capped at x% value"""
    arr2d_rank = np.array(pd.DataFrame(arr2d).rank(axis, method='first'))
    n_vals = np.nanmax(arr2d_rank, axis)
    top_cap = arr2d[arr2d_rank == np.expand_dims(((n_vals - 1) * (1 - pct) + 1).astype('int'), axis)]
    ret = arr2d.copy()
    top_cap_extended = np.full(arr2d.shape[1 - axis], np.nan)
    non_nan_mask = ((~np.isnan(arr2d)).sum(axis) != 0)  # row not all nans
    top_cap_extended[non_nan_mask] = top_cap
    top_cap_extended = np.expand_dims(top_cap_extended, axis)
    if nan == False:
        top_cap_repeated = top_cap_extended.repeat(arr2d.shape[axis], axis)
        ret[ret > top_cap_extended] = top_cap_repeated[ret > top_cap_extended]
    else:
        ret[ret > top_cap_extended] = np.nan
    return ret


def bottom_cap(arr2d, pct, axis=1, nan=False):
    """bottom x% are capped at x% value"""
    arr2d_rank = np.array(pd.DataFrame(arr2d).rank(axis, method='first'))
    n_vals = np.nanmax(arr2d_rank, axis)
    bottom_cap = arr2d[arr2d_rank == np.expand_dims(((n_vals - 1) * pct + 1).astype('int'), axis)]
    ret = arr2d.copy()
    bottom_cap_extended = np.full(arr2d.shape[1 - axis], np.nan)
    non_nan_mask = ((~np.isnan(arr2d)).sum(axis) != 0)  # row not all nans
    bottom_cap_extended[non_nan_mask] = bottom_cap
    bottom_cap_extended = np.expand_dims(bottom_cap_extended, axis)
    if nan == False:
        bottom_cap_repeated = bottom_cap_extended.repeat(arr2d.shape[axis], axis)
        ret[ret < bottom_cap_extended] = bottom_cap_repeated[ret < bottom_cap_extended]
    else:
        ret[ret < bottom_cap_extended] = np.nan
    return ret


def top_bottom_only(arr2d, bottom_pct, top_pct, axis=1, equal_weights=False, decay=True):
    """middle values are set to nan"""
    ranks = rank(arr2d, axis=axis, method='ordinal')
    max_rank = np.expand_dims(np.nanmax(ranks, axis), axis) + 1

    mask_top = (ranks >= max_rank * (1 - top_pct))
    mask_bottom = (ranks < max_rank * bottom_pct)

    ret = arr2d.copy()
    # 这步操作会造成信号十分稀疏，若不前值填充则换手过高，若前值填充则换手很低而多空效果消失。作为折衷我们作衰减前值填充。
    if decay == False:
        ret[(~mask_top) & (~mask_bottom)] = np.nan
    else:
        ret[(~mask_top) & (~mask_bottom) & (~np.isnan(arr2d))] = 0  # 第三个mask很关键
    if equal_weights == True:
        ret[mask_bottom] = -1
        ret[mask_top] = 1

    return ret


def top_only(arr2d, top_pct, axis=1, equal_weights=False, decay=True):
    """bottom values are set to nan"""
    ranks = rank(arr2d, axis=axis, method='ordinal')
    max_rank = np.expand_dims(np.nanmax(ranks, axis), axis) + 1
    mask_top = (ranks >= max_rank * (1 - top_pct))
    ret = arr2d.copy()
    # 这步操作会造成信号十分稀疏，若不前值填充则换手过高，若前值填充则换手很低而多空效果消失。作为折衷我们作衰减前值填充。
    if decay == False:
        ret[~mask_top] = np.nan
    else:
        ret[~mask_top & (~np.isnan(arr2d))] = 0
    if equal_weights == True:
        ret[mask_top] = 1
    return ret


def bottom_only(arr2d, bottom_pct, axis=1, equal_weights=False, decay=True):
    """top values are set to nan"""
    ranks = rank(arr2d, axis=axis, method='ordinal')
    max_rank = np.expand_dims(np.nanmax(ranks, axis), axis) + 1
    mask_bottom = (ranks < max_rank * bottom_pct)
    ret = arr2d.copy()
    # 这步操作会造成信号十分稀疏，若不前值填充则换手过高，若前值填充则换手很低而多空效果消失。作为折衷我们作衰减前值填充。
    if decay == False:
        ret[~mask_bottom] = np.nan
    else:
        ret[~mask_bottom & (~np.isnan(arr2d))] = 0
    if equal_weights == True:
        ret[mask_bottom] = -1
    return ret


def nan_equal(a, b):
    """Given two objects, check that all elements of these objects are equal."""
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def filter_extreme_MAD(arr2d, axis=1, n=5):
    median = np.nanquantile(arr2d, q=0.5, axis=axis)
    MAD = np.nanquantile(np.abs(arr2d - np.expand_dims(median, axis)), q=0.5, axis=axis)  # mean absolute deviance
    max_range = np.expand_dims(median + n * MAD, axis=axis)
    min_range = np.expand_dims(median - n * MAD, axis=axis)
    return np.clip(arr2d, min_range, max_range)


def calPearsonR(x, y, axis=1):
    """calculate pearsonr along an axis."""
    x = x - np.expand_dims(np.nanmean(x, axis=1), 1)
    y = y - np.expand_dims(np.nanmean(y, axis=1), 1)
    mask = (~np.isnan(x) & ~np.isnan(y))
    nanmask = (np.isnan(x) | np.isnan(y))  # make x and y have the same nan values
    x[nanmask] = np.nan
    y[nanmask] = np.nan
    result = np.nansum(x * y, 1) / np.sqrt(np.nansum(x ** 2, 1) * np.nansum(y ** 2, 1))
    result[mask.sum(axis=axis) <= 100] = np.nan  # at least 100 stocks to compute IC
    return result


def rank(arr, axis=1, method='average'):
    """rank along an axis, starting at zero"""
    from scipy import stats
    ranks = stats.rankdata(arr, method=method, axis=axis).astype('float')  # nans are given largest rank
    ranks[np.isnan(arr)] = np.nan  # mstats.rankdata assign 0 to masked values
    return ranks - 1


def rolling_inflated_array(arr, window, axis=0):
    """
    We include some source codes to make this compatible with NumPy 1.19
    inflate an array, where the new dimension is all past values along the rolling axis, inserted at the end.
    e.g. np.nanmax(rolling_inflated_array(arr, 5, axis=0), axis=-1) is equivalent to pd.rolling(pd.DataFrame(arr), axis=0, window=5).max()
    This is especially convenient for rolling argmax operations.
    """

    def sliding_window_view(x, window_shape, axis=None, *,
                            subok=False, writeable=False):
        import numpy as np
        from numpy.core.numeric import normalize_axis_tuple
        from numpy.core.overrides import array_function_dispatch, set_module
        from numpy.lib.stride_tricks import as_strided

        window_shape = (tuple(window_shape)
                        if np.iterable(window_shape)
                        else (window_shape,))
        # first convert input to array, possibly keeping subclass
        x = np.array(x, copy=False, subok=subok)

        window_shape_array = np.array(window_shape)
        if np.any(window_shape_array < 0):
            raise ValueError('`window_shape` cannot contain negative values')

        if axis is None:
            axis = tuple(range(x.ndim))
            if len(window_shape) != len(axis):
                raise ValueError(f'Since axis is `None`, must provide '
                                 f'window_shape for all dimensions of `x`; '
                                 f'got {len(window_shape)} window_shape elements '
                                 f'and `x.ndim` is {x.ndim}.')
        else:
            axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
            if len(window_shape) != len(axis):
                raise ValueError(f'Must provide matching length window_shape and '
                                 f'axis; got {len(window_shape)} window_shape '
                                 f'elements and {len(axis)} axes elements.')

        out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

        # note: same axis can be windowed repeatedly
        x_shape_trimmed = list(x.shape)
        for ax, dim in zip(axis, window_shape):
            if x_shape_trimmed[ax] < dim:
                raise ValueError(
                    'window shape cannot be larger than input array shape')
            x_shape_trimmed[ax] -= dim - 1
        out_shape = tuple(x_shape_trimmed) + window_shape
        return as_strided(x, strides=out_strides, shape=out_shape,
                          subok=subok, writeable=writeable)

    fill_shape = list(arr.shape)
    fill_shape[axis] = window - 1
    arr_concatenated = np.concatenate([np.full(fill_shape, np.nan), arr], axis=axis)  # add some nans before
    return np.swapaxes(sliding_window_view(arr_concatenated, window_shape=arr.shape[0], axis=axis), 0, -1)


def wraped_nanfuncs(f, arr, axis):
    """wrap for funcs such as np.nanargmin, which raises exception when encoutering all-nan row/..."""
    ALL_NAN_MASK = ((~np.isnan(arr)).sum(axis) == 0)  # values along axis=axis are not all nans
    arr = np.where(np.isnan(arr), np.inf, arr)
    ret = f(arr, axis).astype('float')
    ret[ALL_NAN_MASK] = np.nan
    return ret


def remove_inf(arr):
    """"""
    return np.where(np.isinf(arr), np.nan, arr)
