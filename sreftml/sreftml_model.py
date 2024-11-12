import copy
import itertools

import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import tensorflow as tf
from sklearn.model_selection import GroupKFold
import datetime
from . import utilities
import os


class SReFT(tf.keras.Model):
    """
    A model class that extends tf.keras.Model for SReFT_ML.

    Attributes:
        activation (str): The activation function to use.
        activation_offsetT (str): The activation function for offsetT.
        output_dim (int): The dimension of the output.
        latent_dim (int): The dimension of the latent variable.
        offsetT_min (float): The minimum value of offsetT.
        offsetT_max (float): The maximum value of offsetT.
        lnvar_y (tf.Variable): The lnvar_y variable.
        model_1 (tf.keras.Sequential): A keras model for estimating offsetT.
        model_y (tf.keras.Sequential): A keras model for estimating prediction.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim_model_1: int,
        latent_dim_model_y: int,
        activation_model_1_mid: str = "sigmoid",
        activation_model_1_out: str = "softplus",
        activation_model_y_mid: str = "tanh",
        offsetT_min: float = -np.inf,
        offsetT_max: float = np.inf,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize a new instance of SReFT_ML.

        Args:
            output_dim (int, optional): The dimension of the output. Defaults to 4.
            latent_dim_model_1 (int): The dimension of the latent dimention of model_1.
            latent_dim_model_1 (int): The dimension of the latent dimention of model_y.
            activation_model_1_mid (str, optional): The activation function to use. Defaults to "sigmoid".
            activation_model_1_out (str, optional): The activation function to use. Defaults to "softplus".
            activation_model_y_mid (str, optional): The activation function to use. Defaults to "tanh".
            offsetT_min (float, optional): The minimum value of offsetT. Defaults to -np.inf.
            offsetT_max (float, optional): The maximum value of offsetT. Defaults to np.inf.
            random_state (int | None, optional): The seed for random number generation. Defaults to None.
        """
        super(SReFT, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform(seed=random_state)
        tf.random.set_seed(random_state)

        self.output_dim = int(output_dim)
        self.latent_dim_model_1 = int(latent_dim_model_1)
        self.latent_dim_model_y = int(latent_dim_model_y)
        self.activation_model_1_mid = activation_model_1_mid
        self.activation_model_1_out = activation_model_1_out
        self.activation_model_y_mid = activation_model_y_mid

        self.offsetT_min = offsetT_min
        self.offsetT_max = offsetT_max

        self.lnvar_y = tf.Variable(tf.zeros(self.output_dim))

        self.model_1 = tf.keras.Sequential(name="estimate_offsetT")
        self.model_1.add(
            tf.keras.layers.Dense(
                self.latent_dim_model_1,
                activation=self.activation_model_1_mid,
                kernel_initializer=initializer,
            )
        )
        self.model_1.add(
            tf.keras.layers.Dense(
                1,
                activation=self.activation_model_1_out,
                kernel_initializer=initializer,
            )
        )

        self.model_y = tf.keras.Sequential(name="estimate_prediction")
        self.model_y.add(
            tf.keras.layers.Dense(
                self.latent_dim_model_y,
                activation=self.activation_model_y_mid,
                kernel_initializer=initializer,
            )
        )
        self.model_y.add(
            tf.keras.layers.Dense(
                self.output_dim, activation=None, kernel_initializer=initializer
            )
        )
        self.random_state = random_state

    def call(
        self,
        inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """
        Call the model with the given inputs.

        Args:
            inputs (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): The inputs for the model.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: The predicted y values.
            :param **kwargs:
        """
        (input_x, input_cov, input_m, input_y) = inputs
        input1 = tf.concat((input_m, input_cov), axis=-1, name="concat")
        offset = self.model_1(input1, training=training)
        offset = tf.clip_by_value(
            offset, self.offsetT_min, self.offsetT_max, name="clip"
        )
        dis_time = tf.add(input_x, offset, name="add")

        input2 = tf.concat((dis_time, input_cov), axis=-1, name="concat")
        y_pred = self.model_y(input2, training=training)

        obj = utilities.tf_compute_negative_log_likelihood(
            input_y, y_pred, self.lnvar_y
        )
        self.add_loss(tf.reduce_sum(obj))
        self.add_metric(tf.reduce_mean(obj), name="loss")

        return y_pred

    def build_graph(self, shapes: tuple[int, int, int, int]) -> tf.keras.Model:
        """
        Build the computational graph for the model.

        Args:
            shapes (tuple[int, int, int, int]): The shapes of the inputs.

        Returns:
            tf.keras.Model: The model with the built computational graph.
        """
        input_x = tf.keras.layers.Input(shape=shapes[0], name="time")
        input_cov = tf.keras.layers.Input(shape=shapes[1], name="covariate")
        input_m = tf.keras.layers.Input(shape=shapes[2], name="feature")
        input_y = tf.keras.layers.Input(shape=shapes[3], name="observation")
        return tf.keras.Model(
            inputs=[input_x, input_cov, input_m, input_y],
            outputs=self.call((input_x, input_cov, input_m, input_y)),
        )

    def get_config(self):
        config = super(SReFT, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'latent_dim_model_1': self.latent_dim_model_1,
            'latent_dim_model_y': self.latent_dim_model_y,
            'activation_model_1_mid': self.activation_model_1_mid,
            'activation_model_1_out': self.activation_model_1_out,
            'activation_model_y_mid': self.activation_model_y_mid,
            'offsetT_min': self.offsetT_min,
            'offsetT_max': self.offsetT_max,
            'random_state': self.random_state
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def hp_search_for_sreftml(
    df: pd.DataFrame,
    scaled_features: tuple,
    grid_dict: dict[list[any]],
    n_grid_sample: int = 0,
    n_splits: int = 3,
    random_seed: int = 42,
    callbacks: list[any] | None = None,
    epochs: int = 9999,
    batch_size: int = 256,
    save_dir: str = "/Users/tamutomo/Library/CloudStorage/OneDrive-千葉大学/lab/SReFT/ROOT/output/Grid"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform hyperparameter search for the SReFT_ML with detailed iteration tracking.

    Args:
        ...

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - df_grid: DataFrame with scores for all iterations.
            - df_min_scores: DataFrame with the minimum score for each hyperparameter combination and its iteration.
    """
    proc_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")[2:]
    base_save_dir = save_dir + "/" + proc_time
    os.mkdir(base_save_dir)
    
    def generate_dir_name(grid_items):
        param_str = "_".join([f"{val}" for key, val in grid_items.items()])
        return param_str
    
    grid_prms = [i for i in itertools.product(*grid_dict.values())]
    df_grid = pd.DataFrame(grid_prms, columns=grid_dict.keys())
    if n_grid_sample > 0:
        df_grid = df_grid.sample(min(int(n_grid_sample), len(df_grid)))

    x_scaled, cov_scaled, m_scaled, y_scaled = scaled_features

    # 評価指標の格納用に拡張
    scores = []
    iterations = []
    params = []
    within_combo_iter_counts = []  # 組み合わせごとにリセットされるイテレーション番号用
    gkf = GroupKFold(n_splits=n_splits)
    
    total_iterations = len(df_grid) * n_splits  # 全体のイテレーション数
    current_total_iter = 1  # 総イテレーションカウンタ
    
    for tmp_grid, grid_items in df_grid.iterrows():
        folder_name = generate_dir_name(grid_items)
        hp_dir = os.path.join(base_save_dir, folder_name)
        os.makedirs(hp_dir, exist_ok=True)
        
        # 各ハイパーパラメータ組み合わせでのイテレーション番号をリセット
        within_combo_iter = 1  # 組み合わせごとのイテレーションカウントをリセット
        
        for i, (tmp_train_idx, tmp_vali_idx) in enumerate(gkf.split(X=df, groups=df.ID)):
            split_dir = os.path.join(hp_dir, f"split_{i + 1}")
            os.makedirs(split_dir, exist_ok=True)
            
            # メッセージの作成
            current_hp = ", ".join([f"{key}={val}" for key, val in grid_items.items()])
            print(
                f"----------------[ Iteration {current_total_iter} of {total_iterations} ]----------------------\n"
                f"| Combination {tmp_grid + 1} of {len(df_grid)} | Fold {within_combo_iter} of {n_splits} |\n"
                f"| Parameters: {current_hp:<60}|\n"
                "-----------------------------------------------------------------"
            )

            # モデルの訓練やスコア計算などのコード...

            current_total_iter += 1  # 総イテレーションカウンタの更新

            tmp_sreft = SReFT(
                output_dim=y_scaled.shape[1],
                latent_dim_model_1=m_scaled.shape[1],
                latent_dim_model_y=y_scaled.shape[1],
                activation_model_1_mid=grid_items["activation_model_1_mid"],
                activation_model_1_out=grid_items["activation_model_1_out"],
                activation_model_y_mid=grid_items["activation_model_y_mid"],
                random_state=random_seed,
            )
            tmp_sreft.compile(optimizer=tf.keras.optimizers.Adam(grid_items["adam_lr"]))
            tmp_sreft.fit(
                (
                    x_scaled[tmp_train_idx, :],
                    cov_scaled[tmp_train_idx, :],
                    m_scaled[tmp_train_idx, :],
                    y_scaled[tmp_train_idx, :],
                ),
                y_scaled[tmp_train_idx, :],
                validation_data=(
                    (
                        x_scaled[tmp_vali_idx, :],
                        cov_scaled[tmp_vali_idx, :],
                        m_scaled[tmp_vali_idx, :],
                        y_scaled[tmp_vali_idx, :],
                    ),
                    y_scaled[tmp_vali_idx, :],
                ),
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                callbacks=callbacks,
            )
            
            # モデルとテストデータの保存
            model_path = os.path.join(split_dir, "sreft_model")
            tmp_sreft.save(model_path, save_format="tf")

            test_data_df = df.iloc[tmp_vali_idx]
            test_data_path = os.path.join(split_dir, "test_data_from_df.csv")
            test_data_df.to_csv(test_data_path, index=False)
            
            # スコア計算と保存
            y_pred = tmp_sreft(
                (
                    x_scaled[tmp_vali_idx, :],
                    cov_scaled[tmp_vali_idx, :],
                    m_scaled[tmp_vali_idx, :],
                    y_scaled[tmp_vali_idx, :],
                )
            )
            temp_score = utilities.np_compute_negative_log_likelihood(
                y_scaled[tmp_vali_idx, :], y_pred, tmp_sreft.lnvar_y
            )
            score_mean = np.nanmean(temp_score)
            scores.append(score_mean)  # スコアを格納
            iterations.append(within_combo_iter)  # 組み合わせごとにリセットされるイテレーション番号を格納
            within_combo_iter_counts.append(within_combo_iter)  # グリッド全体に対するイテレーション番号
            
            params.append(grid_items)  # 現在のパラメータ設定を格納
            
            within_combo_iter += 1  # 組み合わせ内イテレーションカウントをインクリメント
    
    # df_gridを更新して、各イテレーションごとのスコアとイテレーション番号を含める
    expanded_grid = pd.DataFrame(params).reset_index(drop=True)
    expanded_grid["score"] = scores
    expanded_grid["combo_iter"] = within_combo_iter_counts  # 各組み合わせ内でのイテレーション番号を追加

    # 各組み合わせごとの最低スコアを取得し、該当するイテレーション番号も含める
    min_score_df = expanded_grid.loc[expanded_grid.groupby(list(grid_dict.keys()))["score"].idxmin()]
    expanded_grid.to_csv(f"{base_save_dir}/df_grid.csv")
    min_score_df.to_csv(f"{base_save_dir}/min_score_df.csv")
    return expanded_grid, min_score_df
