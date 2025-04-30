"""
utilities.py
※ データ整形、スケーリング、モデル学習、評価指標など SReFT-ML に必要な汎用的な関数群を提供します。
"""

import os
os.chdir("/Users/tamutomo/OneDrive - 千葉大学/lab/SReFT/ROOT")
import math
import pickle
import subprocess
import autograd.numpy as agnp
import lifelines
import numpy as np
import pandas as pd
import shap
import csv
import shutil
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import statsmodels.formula.api as smf
import tensorflow as tf
from tensorflow import keras
import statistics
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from src.sreft_ml_cpp.system import config
from src.sreft_ml_cpp.system.log_utils import dprint
import warnings
import getpass
warnings.filterwarnings("ignore")
class NullModel:
    def __init__(self, Intercept, TIME):
        self.params = [Intercept, TIME]


def n2mfrow(n_plots: int, ncol_max: int = 4) -> tuple[int, int]:
    """
    Determines the number of rows and columns required to plot a given number of subplots.

    Args:
        n_plots (int): Total number of subplots.
        ncol_max (int, optional): Maximum number of columns for subplots. Defaults to 4.

    Returns:
        tuple: (number of rows, number of columns)"""
    n_plots = int(n_plots)
    nrow = math.ceil(n_plots / ncol_max)
    ncol = math.ceil(n_plots / nrow)
    return nrow, ncol


def linear_regression_each_subject(
    df: pd.DataFrame, y_columns: list[str]
) -> pd.DataFrame:
    """
    Perform linear regression for each subject (ID) in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for the regression. It must include columns for 'ID', 'TIME', and the target variables specified in 'y_columns'.
        y_columns (list[str]): A list of column names (strings) representing the target variables to be regressed.

    Returns:
        pd.DataFrame: A DataFrame with the regression results for each subject.
    """
    model = LinearRegression()
    results = {"ID": df.ID.unique()}

    for y in y_columns:
        slopes = []
        intercepts = []

        for _, group in df.groupby("ID"):
            x_values = group["TIME"].values.reshape(-1, 1)
            y_values = group[y].values

            valid_mask = ~np.isnan(y_values)
            valid_sample_count = valid_mask.sum()

            if valid_sample_count == 0:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                continue

            model.fit(x_values[valid_mask], y_values[valid_mask])

            if valid_sample_count == 1:
                slopes.append(np.nan)
            else:
                slopes.append(model.coef_[0])
            intercepts.append(model.intercept_)
        results[f"{y}_slope"] = slopes
        results[f"{y}_intercept"] = intercepts

    result = pd.DataFrame(results)
    result = result[
        ["ID"] + [i + j for j in ["_slope", "_intercept"] for i in y_columns]
    ]

    return result


def mixed_effect_linear_regression(
    df: pd.DataFrame, y_columns: list[str]
) -> tuple[pd.DataFrame, list]:
    """
    Perform mixed-effects linear regression on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for the regression.
            It must include columns for 'ID', 'TIME', and the target variables specified in 'y_columns'.
        y_columns (list[str]): A list of column names (strings) representing the target variables to be regressed.

    Returns:
        tuple: A tuple containing two elements:
            - result (pd.DataFrame): The DataFrame with the fitted regression parameters for each individual.
            - models (list): A list of fitted mixed-effects regression models for each target variable.
    """
    result = pd.DataFrame(df.ID.unique()).set_axis(["ID"], axis=1)
    models = []

    for y in y_columns:
        df_ = (
            df[["ID", "TIME", y]]
            .dropna()
            .reset_index(drop=True)
            .set_axis(["ID", "TIME", "TARGET"], axis=1)
        )
        if df_["TIME"].nunique() == 1:
            warnings.warn(
                f"Only one time point is available for {y}. The slope cannot be calculated."
            )
            tmp = pd.DataFrame(
                {
                    "ID": df_.ID.unique(),
                    f"{y}_slope": np.nan,
                    f"{y}_intercept": df_.groupby("ID")["TARGET"].mean().values,
                }
            )
            result = result.merge(tmp, how="outer")
            models.append(NullModel(df_.groupby("ID")["TARGET"].mean().mean(), np.nan))
            continue

        full_model = smf.mixedlm(
            "TARGET ~ TIME", data=df_, groups="ID", re_formula="~TIME"
        ).fit()
        random_effects = pd.DataFrame(full_model.random_effects).T.values
        params_pop = full_model.params[0:2].values.T
        params_ind = pd.DataFrame(params_pop + random_effects).set_axis(
            [f"{y}_intercept", f"{y}_slope"], axis=1
        )
        params_ind["ID"] = pd.DataFrame(full_model.random_effects).T.index.values
        result = result.merge(params_ind, how="outer")
        models.append(full_model)

    result = result[
        ["ID"] + [i + j for j in ["_slope", "_intercept"] for i in y_columns]
    ]
    return result, models


def split_data_for_sreftml(
    df: pd.DataFrame,
    name_biomarkers: list[str],
    name_covariates: list[str],
    isMixedlm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data for sreftml.

    Args:
        df (pd.DataFrame): Input DataFrame.
        name_biomarkers (list[str]): List of biomarker names.
        name_covariates (list[str]): List of covariate names.
        isMixedlm (bool): Select whether to use a mixed-effects model when computing model_1 features. Default to True.

    Returns:
        tuple: A tuple containing the following arrays:
            - x (pd.DataFrame): Time values.
            - cov (pd.DataFrame): Covariate values.
            - m (pd.DataFrame): Slope and intercept from regression by biomarker.
            - y (pd.DataFrame): Biomarker values.
    """
    df_ = df.copy()
    if len(name_covariates) > 0 and pd.isna(df[name_covariates]).any().any():
        warnings.warn("Missing value imputation was performed for some covariates.")
        df_[name_covariates] = df_[name_covariates].fillna(
            df.loc[:, name_covariates].mean()
        )

    if isMixedlm:
        linreg, models = mixed_effect_linear_regression(df_, name_biomarkers)
        if pd.isna(linreg).any().any():
            warnings.warn("Missing value imputation was performed for some features.")
            prms = [i.params[0] for i in models] + [i.params[1] for i in models]
            labels = [i + j for j in ["_intercept", "_slope"] for i in name_biomarkers]
            dict_slope = dict(zip(labels, prms))
            linreg = linreg.fillna(dict_slope)
    else:
        linreg = linear_regression_each_subject(df_, name_biomarkers)
        if pd.isna(linreg).any().any():
            warnings.warn("Missing value imputation was performed for some features.")
            linreg = linreg.fillna(linreg.mean())
            
    df_ = df_.merge(linreg)

    x = df_.TIME
    cov = df_[name_covariates]
    m = df_.loc[:, df_.columns.str.contains("_slope|_intercept")].dropna(
        axis=1, how="all"
    )
    y = df_[name_biomarkers]

    return x, cov, m, y, linreg


def np_compute_negative_log_likelihood(
    y_true: np.ndarray, y_pred: np.ndarray, lnvar_y: np.ndarray
) -> np.ndarray:
    """
    Computes the negative log likelihood between true and predicted values using numpy.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.
        lnvar_y (np.array): Natural logarithm of the variance.

    Returns:
        np.array: The negative log likelihood for each instance.
    """
    neg_ll = lnvar_y + np.power(y_true - y_pred, 2) / np.exp(lnvar_y)
    return np.nansum(neg_ll, axis=1)


def tf_compute_negative_log_likelihood(
    y_true: np.ndarray, y_pred: np.ndarray, lnvar_y: tf.Variable
) -> tf.Tensor:
    """
    Computes the negative log likelihood between true and predicted values using tensorflow.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        lnvar_y (tf.Variable): Natural logarithm of the variance.

    Returns:
        tf.Tensor: The negative log likelihood for each instance.
    """
    is_nan = tf.math.is_nan(y_true)
    y_true = tf.where(is_nan, tf.zeros_like(y_true), y_true)
    y_pred = tf.where(is_nan, tf.zeros_like(y_pred), y_pred)
    neg_ll = lnvar_y + tf.pow(y_true - y_pred, 2) / tf.exp(lnvar_y)
    neg_ll = tf.where(is_nan, tf.zeros_like(neg_ll), neg_ll)

    return tf.reduce_sum(neg_ll, axis=1)


class DummyTransformer:
    def __init__(
        self,
    ):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X


def get_current_commit_hash() -> str:
    """
    Retrieves the current commit hash of the git repository.

    Returns:
        str: The current commit hash or a placeholder string if an error occurs.
    """
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return commit_hash.strip().decode("utf-8")
    except subprocess.CalledProcessError:
        warnings.warn("Could not get the current commit hash.", UserWarning)
        return "commit_hash_not_available"


def clean_duplicate(
    df: pd.DataFrame, cols: list[str], duplicate_key: list[str] | str | None
) -> pd.DataFrame:
    """
    Checks for duplicate entries in the DataFrame based on the specified columns and removes NaNs; also removes duplicate entries if a subset is specified.

    Parameters:
        df (pd.DataFrame): The DataFrame to check and drop duplicates from.
        cols (list[str]): List of column names to check (and remove) for duplicates.
        duplicate_key (list[str] | str | None): If specify, duplicate deletion will be performed. Then, check duplicate within sepecified columns.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed. It includes only the columns specified in cols and duplicate_key.

    Warnings:
        If any duplicates are found in the DataFrame after cleaning, a warning message is displayed.
        The warning message depends on the `subset` parameter:
        - If `subset` is None, the warning message indicates that some records are duplicates across all columns in `cols`.
        - If `subset` is not None, the warning message indicates that some records are duplicates within the same subset.
    """
    if type(duplicate_key) is str:
        duplicate_key = [duplicate_key]

    if duplicate_key is None:
        df_ = df[cols].dropna()
        if df_.duplicated().any():
            warnings.warn(
                "Some records are duplicates. Set duplicate_key if necessary."
            )
    else:
        df_ = df[cols + duplicate_key].dropna().drop_duplicates()
        if df_.duplicated(subset=duplicate_key).any():
            warnings.warn(
                "Duplicate records remain in some duplicate_keys. Add duplicate_key if necessary."
            )

    return df_


def compute_permutation_importance_(
    random_seed: int,
    sreft: tf.keras.Model,
    x_test: np.ndarray,
    cov_test: np.ndarray,
    m_test: np.ndarray,
    y_test: np.ndarray,
    n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [Superseded] Compute permutation importance of the model.

    Args:
        random_seed (int): The seed for the random number generator.
        sreft (tf.keras.Model): The model for which to calculate permutation importance.
        x_test (np.ndarray): The x test data.
        cov_test (np.ndarray): The covariates test data.
        m_test (np.ndarray): The m test data.
        y_test (np.ndarray): The y test data.
        n_sample (int): The number of samples.

    Returns:
        tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the permutation importance.
    """
    rng = np.random.default_rng(random_seed)
    y_pred = sreft((x_test, cov_test, m_test, y_test)).numpy()
    neglls_orig = np_compute_negative_log_likelihood(y_test, y_pred, sreft.lnvar_y)

    mean_pi = []
    std_pi = []
    n_pi = m_test.shape[1] + cov_test.shape[1]

    for i in range(n_pi):
        pis = []
        for j in range(n_sample):
            if i < m_test.shape[1]:
                m_test_rand = np.copy(m_test)
                rng.shuffle(m_test_rand[:, i])
                y_pred_rand = sreft((x_test, cov_test, m_test_rand, y_test)).numpy()
            else:
                cov_test_rand = np.copy(cov_test)
                rng.shuffle(cov_test_rand[:, i - m_test.shape[1]])
                y_pred_rand = sreft((x_test, cov_test_rand, m_test, y_test)).numpy()

            neglls_rand = np_compute_negative_log_likelihood(
                y_test, y_pred_rand, sreft.lnvar_y
            )
            nglls_diff = neglls_rand - neglls_orig
            temp_pi = np.nanmean(nglls_diff)
            pis.append(temp_pi)

        mean_pi.append(np.mean(pis))
        std_pi.append(np.std(pis))

    return np.array(mean_pi), np.array(std_pi)


def compute_permutation_importance(
    random_seed: int,
    sreft: tf.keras.Model,
    cov_test: np.ndarray,
    m_test: np.ndarray,
    n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute permutation importance of the model.

    Args:
        random_seed (int): The seed for the random number generator.
        sreft (tf.keras.Model): The model for which to calculate permutation importance.
        cov_test (np.ndarray): The covariates test data.
        m_test (np.ndarray): The m test data.
        n_sample (int): The number of samples.

    Returns:
        tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the permutation importance.
    """
    rng = np.random.default_rng(random_seed)
    offestt_pred = sreft.model_1(np.concatenate((m_test, cov_test), axis=-1)).numpy()

    mean_pi = []
    std_pi = []
    n_pi = m_test.shape[1] + cov_test.shape[1]

    for i in range(n_pi):
        pis = []
        for j in range(n_sample):
            if i < m_test.shape[1]:
                m_test_rand = np.copy(m_test)
                rng.shuffle(m_test_rand[:, i])
                y_pred_rand = sreft.model_1(
                    np.concatenate((m_test_rand, cov_test), axis=-1)
                ).numpy()
            else:
                cov_test_rand = np.copy(cov_test)
                rng.shuffle(cov_test_rand[:, i - m_test.shape[1]])
                y_pred_rand = sreft.model_1(
                    np.concatenate((m_test, cov_test_rand), axis=-1)
                ).numpy()

            nglls_diff = (offestt_pred - y_pred_rand) ** 2
            temp_pi = np.nanmean(nglls_diff)
            pis.append(temp_pi)

        mean_pi.append(np.mean(pis))
        std_pi.append(np.std(pis))

    return np.array(mean_pi), np.array(std_pi)


def calculate_offsetT_prediction(
    sreft: tf.keras.Model,
    df: pd.DataFrame,
    scaled_features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scaler_y: sp.StandardScaler,
    name_biomarkers: list[str],
    getOffsetT: bool = True,
    getPrediction: bool = True,
) -> pd.DataFrame:
    """
    Calculate offsetT and prediction value of biomarkers.

    Args:
        sreft (tf.keras.Model): The trained SReFT model.
        df (pd.DataFrame): The input DataFrame.
        scaled_features (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): The scaled features. Pass x, cov, m, and y in that order.
        scaler_y (sp.StanderdScaler): The scaler for y.
        name_biomarkers (list[str]): List of biomarker names.

    Returns:
        pd.DataFrame: The DataFrame including the columns of the input DataFrame, offsetT and the prediction values.
    """
    df_ = df.copy()
    x_scaled, cov_scaled, m_scaled, y_scaled = scaled_features

    if getOffsetT:
        offsetT = sreft.model_1(np.concatenate((m_scaled, cov_scaled), axis=-1))
        df_ = df_.reset_index(drop=True).assign(offsetT=offsetT)

    if getPrediction:
        y_pred = pd.DataFrame(
            scaler_y.inverse_transform(sreft(scaled_features)),
            columns=[f"{biomarker}_pred" for biomarker in name_biomarkers],
        )
        df_ = df_.reset_index(drop=True).assign(**y_pred)

    return df_


class GompertzFitter(lifelines.fitters.ParametricUnivariateFitter):
    _fitted_parameter_names = ["lambda_", "c_"]

    def _cumulative_hazard(self, params, times):
        lambda_, c_ = params
        return lambda_ / c_ * (agnp.expm1(times * c_))


def survival_analysis(
    df: pd.DataFrame,
    surv_time: str,
    event: str,
    useOffsetT: bool = True,
    gompertz_init_params: list = [0.1, 0.1],
) -> dict:
    """
    Perform survival analysis and return a dictionary of survival analysis objects.


    If the survival time contains 0 or less, the survival time is converted so that the minimum value is 0.00001.

    Args:
        df (pd.DataFrame): Input DataFrame.
        surv_time (str): Column name of the survival time in df.
        event (str): Column name of the event in df.
        useOffsetT (bool, optional): Determines whether to use offsetT for the analysis. Defaults to True.

    Returns:
        dict: A dictionary of survival analysis objects.
    """
    fitters = [
        (lifelines.KaplanMeierFitter, "kmf", "KaplanMeier"),
        (lifelines.NelsonAalenFitter, "naf", "NelsonAalen"),
        (lifelines.ExponentialFitter, "epf", "Exponential"),
        (lifelines.WeibullFitter, "wbf", "Weibull"),
        (GompertzFitter, "gpf", "Gompertz"),
        (lifelines.LogLogisticFitter, "llf", "LogLogistic"),
        (lifelines.LogNormalFitter, "lnf", "LogNormal"),
    ]
    fit_model = {"title": event}
    if useOffsetT:
        df_surv = df[["ID", "offsetT", surv_time, event]].dropna().drop_duplicates()

        if df_surv["offsetT"].min() < 0:
            raise ValueError("offsetT must be greater than or equal to 0.")

        for fitter_class, key, label in fitters:
            if key == "gpf":
                fit_model[key] = fitter_class(label=label).fit(
                    durations=df_surv["offsetT"] + df_surv[surv_time],
                    event_observed=df_surv[event],
                    entry=df_surv["offsetT"],
                    initial_point=gompertz_init_params,
                )
            else:
                fit_model[key] = fitter_class(label=label).fit(
                    durations=df_surv["offsetT"] + df_surv[surv_time],
                    event_observed=df_surv[event],
                    entry=df_surv["offsetT"],
                )
    else:
        df_surv = df[["ID", surv_time, event]].dropna().drop_duplicates()
        for fitter_class, key, label in fitters:
            fit_model[key] = fitter_class(label=label).fit(
                durations=df_surv[surv_time], event_observed=df_surv[event]
            )

    return fit_model


def multi_column_filter(
    df: pd.DataFrame,
    upper_lim: dict[str, float] = None,
    lower_lim: dict[str, float] = None,
    IQR_filter: list = None,
):
    """
    Applies limits and IQR filtering on DataFrame columns.

    Operations:
        NaN substitution for values outside the specified upper and lower limits.
        IQR-based outlier removal in specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        upper_lim (dict[str, float], optional): Upper limits per column.
        lower_lim (dict[str, float], optional): Lower limits per column.
        IQR_filter (list, optional): Columns for IQR outlier detection

    Returns:
        pd.DataFrame: DataFrame after applying the defined filters.

    Notes:
        Overlapping `upper_lim`/`lower_lim` and `IQR_filter` keys cause warnings
        and filtering by `upper_lim`/`lower_lim`.
    """
    df_filtered = df.copy()
    if upper_lim is None:
        upper_lim = {}
    if lower_lim is None:
        lower_lim = {}
    if IQR_filter is None:
        IQR_filter = []

    if upper_lim:
        for k, v in upper_lim.items():
            df_filtered.loc[df_filtered[k] > v, k] = np.nan
        overlap_upper_IQR = set(upper_lim.keys()) & set(IQR_filter)
        if overlap_upper_IQR:
            warnings.warn(
                f"The columns {overlap_upper_IQR} were present in both upper_lim and IQR_filter, therefore they were filtered using the values from upper_lim."
            )

    if lower_lim:
        for k, v in lower_lim.items():
            df_filtered.loc[df_filtered[k] < v, k] = np.nan
        overlap_lower_IQR = set(lower_lim.keys()) & set(IQR_filter)
        if overlap_lower_IQR:
            warnings.warn(
                f"The columns {overlap_lower_IQR} were present in both lower_lim and IQR_filter, therefore they were filtered using the values from lower_lim."
            )

    if IQR_filter:
        IQR_exclusive = list(
            set(IQR_filter) - set(upper_lim.keys()) - set(lower_lim.keys())
        )
        q1 = df_filtered.quantile(0.25)
        q3 = df_filtered.quantile(0.75)
        iqr = q3 - q1
        df_filtered[IQR_exclusive] = df_filtered[IQR_exclusive].mask(
            (df_filtered < q1 - 1.5 * iqr) | (df_filtered > q3 + 1.5 * iqr), np.nan
        )

    return df_filtered


def calc_shap_explanation(
    sreft: tf.keras.Model,
    feature_names: list[str],
    cov_scaled: np.ndarray,
    m_scaled: np.ndarray,
) -> shap.Explanation:
    """
    Calculate the SHAP values for model 1.

    Args:
        sreft (tf.keras.Model): The model for which to calculate SHAP values.
        feature_names (list[str]): Provide the column names for 'm' and 'cov'. 'm' comes first, followed by 'cov'.
        cov_scaled (np.ndarray): The scaled covariate values.
        m_scaled (np.ndarray): The scaled m values.

    Returns:
        shap.Explanation: The explanation of SHAP values.
    """
    input1 = np.concatenate((m_scaled, cov_scaled), axis=-1)
    explainer_model_1 = shap.Explainer(
        sreft.model_1,
        input1,
        algorithm="permutation",
        seed=42,
        feature_names=feature_names,
    )
    shap_value_model_1 = explainer_model_1(input1)
    shap_exp_model_1 = shap.Explanation(
        shap_value_model_1.values,
        shap_value_model_1.base_values[0][0],
        shap_value_model_1.data,
        feature_names=feature_names,
    )

    return shap_exp_model_1


def load_shap(
    path_to_shap_file: str,
) -> shap.Explanation:
    """
    Load the specified SHAP binary file and return the SHAP explanations.

    Args:
        path_to_shap_file (str): The path to the SHAP file.

    Returns:
        Explanation: The explanation of SHAP values.
    """
    with open(path_to_shap_file, "rb") as p:
        shap_exp = pickle.load(p)

    return shap_exp


def save_shap(path_to_shap_file: str, shap_exp: shap.Explanation) -> None:
    """
    Save the SHAP explanations to the specified file.

    Parameters:
        path_to_shap_file (str): The path to save the SHAP file.
        shap_exp (shap.Explanation): The SHAP explanations to be saved.

    Returns:
        None
    """
    with open(path_to_shap_file, "wb") as p:
        pickle.dump(shap_exp, p)

    return None

class TimePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
            self.prepare_data()
        else:
            self.df = pd.DataFrame()

    def prepare_data(self):
        if not self.df.empty:
            for column in self.df.select_dtypes(include=['object']).columns:
                self.label_encoders[column] = LabelEncoder()
                self.df[column] = self.label_encoders[column].fit_transform(self.df[column].astype(str))

    def save_data(self):
        self.df.to_csv(self.data_path, index=False)

    def predict(self, initial_data, remaining_iterations, func_list_1, func_list_2, func_list_3, current_iteration):
        if not self.df.empty and len(self.df) > 10:
            # 四分位範囲を計算し、範囲内のデータのみを選択
            q1 = self.df['required_time'].quantile(0.25)
            q3 = self.df['required_time'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_df = self.df[(self.df['required_time'] >= lower_bound) & (self.df['required_time'] <= upper_bound)]

            X = filtered_df.drop('required_time', axis=1)
            y = filtered_df['required_time']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            dprint(f'Mean Squared Error: {mse}')

            df_predictions = pd.DataFrame(columns=['func_1', 'func_2', 'func_3', 'predicted_time'])

            for i in func_list_1:
                for j in func_list_2:
                    for k in func_list_3:
                        new_data = initial_data.copy()
                        new_data.update({"func_1": i, "func_2": j, "func_3": k})
                        
                        new_data_df = pd.DataFrame([new_data])
                        new_data_df = new_data_df.applymap(lambda x: 1 if x is True else (0 if x is False else x))

                        for column in new_data_df.select_dtypes(include=['object']).columns:
                            new_data_df[column] = new_data_df[column].astype(str)
                            if column in self.label_encoders:
                                new_data_df[column] = self.label_encoders[column].transform(new_data_df[column])

                        # 新しいデータの数値型変換処理
                        for column in new_data_df.columns:
                            if column not in self.label_encoders:
                                new_data_df[column] = pd.to_numeric(new_data_df[column], errors='coerce')

                        predicted_required_time = self.model.predict(new_data_df)[0]
                        df_predictions = pd.concat([df_predictions, pd.DataFrame({'func_1': [i], 'func_2': [j], 'func_3': [k], 'predicted_time': [predicted_required_time]})], ignore_index=True)

            # 修正: current_iteration以降の予測値を合計
            remaining_predictions = df_predictions.iloc[current_iteration:]['predicted_time']
            total_predicted_time = remaining_predictions.sum()

            # 修正: 記録する際に活性化関数の組み合わせを正確に反映
            self.save_predictions(df_predictions)

            return total_predicted_time
        else:
            return self.df['required_time'].mean() * remaining_iterations if not self.df.empty else 0

    def save_predictions(self, df_predictions):
        df_predictions.to_csv(self.data_path.replace(".csv", "_predictions.csv"), index=False)

    def add_data(self, new_data):
        new_data_df = pd.DataFrame([new_data])
        self.df = pd.concat([self.df, new_data_df], ignore_index=True)
        self.prepare_data()
        self.save_data()
        
def df_maker(main_settings,
             path_for_original_data: str,
             outliers_remove: bool = False,
             data_randamize: bool = False,
             impute_method: str = None,
             ):
    
    if "GUIDE-IT" in path_for_original_data:
        name_biomarkers = config.name_biomarkers_guide_it
        name_covariates = config.name_covariates_GUIDE_IT
        rename_dict = config.rename_dict_GUIDE_IT
    elif "solvd" in path_for_original_data:
        name_biomarkers = config.name_biomarkers_solvd
        name_covariates = config.name_covariates_solvd
        rename_dict = config.rename_dict_solvd
    elif "best" in path_for_original_data:
        name_biomarkers = config.name_biomarkers_best
        name_covariates = config.name_covariates_best
        rename_dict = config.rename_dict_best
    
    df = pd.read_csv(path_for_original_data)
    df = (
        df.rename(columns=rename_dict)
          .dropna(subset=["ID", "TIME"], how="any")
          .dropna(subset=name_biomarkers, how="all")
          .reset_index(drop=True)
    )
    
    #df[name_biomarkers] = df[name_biomarkers].fillna(df[name_biomarkers].mean())
    
    if outliers_remove:
        df = remove_outliers(df, name_biomarkers)
        dprint("外れ値を除外しました。")
    if main_settings.impute_method != "none":
        for col in name_biomarkers:
            if main_settings.impute_method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif main_settings.impute_method == "distribution":
                df[col] = impute_by_distribution(df[col])
            elif main_settings.impute_method == "zero":
                df[col] = df[col].fillna(0)
            else:
                raise ValueError(f"Unknown impute method: {main_settings.impute_method}")
        dprint(f"{main_settings.impute_method}方式で補完を行いました。")
        
    if data_randamize:
        df, random_state = data_randomize(df)
        dprint("データセットの行の順番をランダムに入れ替えました。")
    return df, name_biomarkers, name_covariates


def remove_outliers(df, name_biomarkers):
    """
    バイオマーカー列の外れ値を中央値±3SDで除外する。

    Parameters:
    - df: pandas.DataFrame（バイオマーカーを含む）
    - name_biomarkers: list[str]（対象のバイオマーカー名）

    Returns:
    - df_out: 外れ値を除去したDataFrame（新しいコピー）
    """
    df_out = df.copy()

    for biomarker in name_biomarkers:
        original_len = len(df_out[biomarker].dropna())
        median_val = df_out[biomarker].median(numeric_only=True)
        std_val = statistics.pstdev(df_out[biomarker].dropna())
        
        condition = (df_out[biomarker] > median_val - std_val * 3) & (df_out[biomarker] < median_val + std_val * 3)
        df_out.loc[~condition, biomarker] = np.nan  # 外れ値を NaN にする

        dprint(f"{median_val - std_val*3:.3f} < {biomarker} < {median_val + std_val*3:.3f}")
        dprint(f"{biomarker} 外れ値 {original_len - condition.sum()} 件削除　　削除後： {condition.sum()} 件\n")
        
    return df_out

def data_randomize(df):
    random_state = np.random.randint(1, 1001)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df, random_state
    
def prepare_scaled_inputs(df, name_biomarkers, name_covariates, isMixedlm=True):
    """
    dfからx, cov, m, yを抽出し、それぞれをスケーリングする関数。
    covariatesが空のときはDummyTransformerで処理する。

    Returns:
        x_scaled, cov_scaled, m_scaled, y_scaled, linreg,
        scaler_cov, scaler_m, scaler_y
    """

    # スケーラー準備
    if len(name_covariates) > 0:
        scaler_cov = StandardScaler().fit(df[name_covariates].values)
    else:
        scaler_cov = DummyTransformer()

    # 分割
    x, cov, m, y, linreg = split_data_for_sreftml(
        df, name_biomarkers, name_covariates, isMixedlm=isMixedlm
    )

    # スケーリング
    x_scaled = x.values.reshape(-1, 1)
    cov_scaled = scaler_cov.transform(cov.values)

    scaler_m = StandardScaler().fit(m.values)
    scaler_y = StandardScaler().fit(y.values)
    m_scaled = scaler_m.transform(m.values)
    y_scaled = scaler_y.transform(y.values)

    return x_scaled, cov_scaled, m_scaled, y_scaled, linreg, scaler_cov, scaler_m, scaler_y, m


def copy_directory(src, dst):
    """指定されたディレクトリをコピー（存在しない場合のみ実行）"""
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        dprint(f"Copied directory to {dst}")
    else:
        dprint(f"Directory already exists: {dst}, skipping copy.")

def copy_file(src, dst):
    """指定されたファイルをコピー（存在しない場合のみ実行）"""
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
        dprint(f"Copied file to {dst}")
    else:
        dprint(f"File already exists: {dst}, skipping copy.")

def save_execution_code(reference_dir):
    """実行コードの保存処理を関数化"""
    sreftml_src = "/Users/tamutomo/OneDrive - 千葉大学/lab/SReFT/ROOT/src/sreft_ml/sreftml"
    sreftml_dst = os.path.join(reference_dir, "sreftml")
    copy_directory(sreftml_src, sreftml_dst)

    sbf_src = "/Users/tamutomo/OneDrive - 千葉大学/lab/SReFT/ROOT/src/sreft_ml_cpp/Search_Best_Function.py"
    sbf_dst = os.path.join(reference_dir, "Search_Best_Function.py")
    copy_file(sbf_src, sbf_dst)

    dprint("実行コードを保存しました。")
    
def train_sreft_model(df, name_biomarkers, name_covariates,
                      x_scaled, cov_scaled, m_scaled, y_scaled,
                      scaler_y, callbacks, epochs, i, j, k,
                      random_seed, output_dir, l, 
                      utilities, sreftml_model, lr=1e-4):
    
    train_idx, vali_idx = next(GroupShuffleSplit(1, test_size=0.1, random_state=random_seed).split(X=df, groups=df.ID))

    sreft = sreftml_model.SReFT(
        output_dim=len(name_biomarkers),
        latent_dim_model_1=m_scaled.shape[1],
        latent_dim_model_y=y_scaled.shape[1],
        activation_model_1_mid=i,
        activation_model_1_out=j,
        activation_model_y_mid=k,
        random_state=random_seed,
    )

    sreft.compile(optimizer=keras.optimizers.Adam(lr))

    sreft.fit(
        (x_scaled[train_idx, :], cov_scaled[train_idx, :], m_scaled[train_idx, :], y_scaled[train_idx, :]),
        y_scaled[train_idx, :],
        batch_size=sum(train_idx),
        validation_data=((x_scaled[vali_idx, :], cov_scaled[vali_idx, :], m_scaled[vali_idx, :], y_scaled[vali_idx, :]), y_scaled[vali_idx, :]),
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
    )

    df = utilities.calculate_offsetT_prediction(
        sreft, df, (x_scaled, cov_scaled, m_scaled, y_scaled), scaler_y, name_biomarkers
    )
    df["offsetT"] = sreft.model_1(np.concatenate((m_scaled, cov_scaled), axis=-1)).numpy()

    return sreft, df, train_idx, vali_idx

def save_offsetT_vs_slope_intercept_plots(df, name_biomarkers, run_save_path, plots_module):
    """
    各バイオマーカーに対して、offsetT と slope / intercept の相関を描画して保存する関数。

    Parameters:
        df (pd.DataFrame): 入力データフレーム（slope/intercept列が含まれている必要あり）
        name_biomarkers (list): 対象とするバイオマーカー名のリスト
        run_save_path (str): 保存先ディレクトリ
        plots_module (module): scatter_plot 関数を含むプロットモジュール（例: plots）
    """
    slope_dir = os.path.join(run_save_path, "offsetT_vs_slope")
    intercept_dir = os.path.join(run_save_path, "offsetT_vs_intercept")
    os.makedirs(slope_dir, exist_ok=True)
    os.makedirs(intercept_dir, exist_ok=True)

    for biomarker in name_biomarkers:
        if biomarker == "左心駆出率":
            continue
        plots_module.single_panel_scatter_plot(
            df,
            x_col="offsetT",
            y_col=f"{biomarker}_slope",
            hue=None,
            duplicate_key="ID",
            density=True,
            identity=True,
            save_file_path=os.path.join(slope_dir, f"{biomarker}_slope_vs_offsetT_scatter.png")
        )
        plots_module.single_panel_scatter_plot(
            df,
            x_col="offsetT",
            y_col=f"{biomarker}_intercept",
            hue=None,
            duplicate_key="ID",
            density=True,
            identity=True,
            save_file_path=os.path.join(intercept_dir, f"{biomarker}_intercept_vs_offsetT_scatter.png")
        )
    
    dprint("各バイオマーカーの slope, intercept と offsetT の相関を保存しました。")
    
def compute_and_save_permutation_importance(
    random_seed,
    sreft_model,
    cov_scaled,
    m_scaled,
    name_covariates,
    m_columns,
    run_save_path,
    plots_module,
    utilities_module,
    reload_images_func=None
):
    """
    Permutation Importance を計算し、プロットとCSV保存を行う関数。

    Parameters:
        random_seed (int): 乱数シード
        sreft_model (keras.Model): 訓練済みの SReFT モデル
        cov_scaled (np.ndarray): 共変量のスケーリング済みデータ
        m_scaled (np.ndarray): m成分のスケーリング済みデータ
        name_covariates (list): 共変量の名前リスト
        m_columns (list): m成分の元のカラム名（df.m.columnsなど）
        run_save_path (str): 保存先ディレクトリ
        plots_module (module): プロット関数を含むモジュール（例：plots）
        utilities_module (module): compute_permutation_importance を含むモジュール
        reload_images_func (function, optional): UI用画像更新関数（なくてもOK）
    """
    # --- 計算 ---
    mean_pi, std_pi = utilities_module.compute_permutation_importance(
        random_seed, sreft_model, cov_scaled, m_scaled, n_sample=100
    )

    # --- プロット保存 ---
    features = list(m_columns) + name_covariates
    plots_module.permutation_importance_plot(
        mean_pi,
        std_pi,
        feature_label=features,
        save_file_path=os.path.join(run_save_path, "permutation_importance.png")
    )
    if reload_images_func:
        reload_images_func()
    dprint("permutation_importance を保存しました。")

    # --- CSV保存 ---
    df_pi = pd.DataFrame({
        "feature": features,
        "mean_pi": mean_pi,
        "std_pi": std_pi
    })
    pi_csv_path = os.path.join(run_save_path, "pi_results.csv")
    df_pi.to_csv(pi_csv_path, index=False)
    dprint(f"[INFO] PI results saved to: {pi_csv_path}")


def initialize_ci_results_csv(run_save_path, ci_results_filename="ci_results.csv"):
    """
    生存解析結果保存用のci_results.csvを初期化（存在しなければ作成）する関数。

    Args:
        run_save_path (str): 各runディレクトリへのパス
        ci_results_filename (str): 保存するcsvファイル名
    """
    parent_dir = os.path.dirname(run_save_path)
    ci_csv_path = os.path.join(parent_dir, ci_results_filename)

    if not os.path.exists(ci_csv_path):
        with open(ci_csv_path, "w", encoding="utf-8") as f:
            f.write("run,c_index\n")
            
            
def perform_survival_analysis_and_record(
    df,
    run_save_path,
    survival_analysis_func,
    surv_plot_func,
    reload_images_func=None,
    concordance_index_func=concordance_index,
    ci_results_filename="ci_results.csv"
):
    """
    offsetTを用いた生存解析とC-indexの記録を行う関数。

    Returns:
        c_index_lifelines (float): 計算されたC-index
    """
    # 必要列を作成
    df["Dead_point_year"] = df["Dead day"] / 365.25
    df["ALL_DEATH"] = df["DEATH"]
    
    
    try:
        fit_model = survival_analysis_func(
            df, surv_time="Dead_point_year", event="ALL_DEATH", useOffsetT=True
        )
        surv_plot_func(
            fit_model,
            ci_show=True,
            only_best=True,
            save_dir_path=os.path.join(run_save_path, "")
        )
        if reload_images_func:
            reload_images_func()
    except Exception as e:
        dprint(f"Survival analysis for ALL_DEATH failed: {e}")
    
        # ❗️失敗した場合も空のci_results.csvを作る
        initialize_ci_results_csv(run_save_path, ci_results_filename)
        parent_dir = os.path.dirname(run_save_path)
        ci_csv_path = os.path.join(parent_dir, ci_results_filename)
        run_number = os.path.basename(run_save_path)
        dprint("C-index_lifelines: 0.000")
        with open(ci_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{run_number},0.000\n")
    
        return 0.000  # 👈ここでC-index=0としてreturnする

    # C-index 計算
    df_c = df[["Dead_point_year", "ALL_DEATH", "offsetT", "ID"]].dropna()
    c_index_lifelines = concordance_index_func(
        df_c['Dead_point_year'], -df_c['offsetT'], df_c['ALL_DEATH']
    )
    dprint(f"C-index_lifelines: {c_index_lifelines}")
    dprint("生存時間解析結果を保存しました。")

    # --- 結果記録 ---
    initialize_ci_results_csv(run_save_path, ci_results_filename)
    parent_dir = os.path.dirname(run_save_path)
    ci_csv_path = os.path.join(parent_dir, ci_results_filename)

    run_number = os.path.basename(run_save_path)
    with open(ci_csv_path, "a", encoding="utf-8") as f:
        f.write(f"{run_number},{c_index_lifelines}\n")

    return c_index_lifelines



def rename_dir(old_dir_path, new_dir_path):
    os.rename(old_dir_path, new_dir_path)

def offsetT_pred_save(df, output_dir):
    df[["ID", "offsetT"]].drop_duplicates("ID").sort_values("ID").to_csv(
        os.path.join(output_dir, "offsetT_pred.csv"), index=False
    )
    return df["offsetT"].mean()
    
def mkdirs(old_dir_path, new_dir_name):
    new_dir_path = f"{old_dir_path}/{new_dir_name}"
    os.makedirs(new_dir_path)
    return new_dir_path

def make_random_numbers(loop_times: int, low: int = 1, high: int = 1000000) -> list[int]:
    """
    指定された回数分のランダムな整数を生成する関数。

    Parameters:
        loop_times (int): 生成する整数の数
        low (int): 最小値（デフォルト: 1）
        high (int): 最大値（デフォルト: 1000000）

    Returns:
        list[int]: ランダムな整数のリスト（重複なし）
    """
    return np.random.choice(range(low, high), size=loop_times, replace=False).tolist()


def get_user_type() -> str:
    """
    実行ユーザーに応じて環境タイプを返す。
    Returns:
        str: "macbook" / "macmini" / "unknown"
    """
    user = getpass.getuser()
    if user in "tamutomo":  # ←ここを書き換えてね！
        return "macbook"
    elif user in "tamura":
        return "macmini"
    else:
        return "unknown"
    
def impute_by_distribution(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    is_na = series.isna()
    n_missing = is_na.sum()
    if n_missing == 0:
        return series
    
    # 元の分布に従った乱数を生成（clipで極端な値は避けてもいい）
    imputed_values = np.random.normal(loc=mean, scale=std, size=n_missing)
    series_filled = series.copy()
    series_filled[is_na] = imputed_values
    return series_filled