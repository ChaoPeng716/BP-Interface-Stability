from pathlib import Path
import numpy as np
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import torch
from chemprop import data, featurizers, models, nn, utils
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging



def configure_silent_mode():
    """Configure silent mode to suppress unnecessary output and warnings."""
    warnings.filterwarnings("ignore")
    
    loggers_to_silence = [
        "pytorch_lightning",
        "pytorch_lightning.accelerators",
        "pytorch_lightning.accelerators.cuda", 
        "pytorch_lightning.accelerators.gpu",
        "pytorch_lightning.core",
        "pytorch_lightning.core.lightning",
        "pytorch_lightning.trainer",
        "pytorch_lightning.trainer.trainer",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.utilities",
        "ray",
        "ray.tune",
        "ray.train"
    ]
    
    for logger_name in loggers_to_silence:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger(logger_name).propagate = False

class DataManager:
    """Handles data loading, preprocessing, splitting, and featurization."""
    
    def __init__(self, df_input: pd.DataFrame, target_label: str = 'Eint_H2O',
                 smiles_label: str = 'CanonSMILES', class_label: str = 'Class_label',
                 ratios: tuple = (0.8, 0.1, 0.1), # (train, val, test)
                 n_splits_cv: int = 10,
                 random_state: int = 123):
        self.df_input = df_input
        self.target_label = target_label
        self.smiles_label = smiles_label
        self.class_label = class_label
        self.ratios = ratios
        self.n_splits_cv = n_splits_cv
        self.random_state = random_state

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.scaler = None
        self.test_mol_ids = None
        self.cv_splits = []

        self._validate_ratios()
        self._prepare_data()

    def _validate_ratios(self):
        if not (len(self.ratios) == 3 and abs(sum(self.ratios) - 1.0) < 1e-9):
            raise ValueError("Ratios must be a tuple of three numbers (train, val, test) that sum to 1.")
        if not all(0 <= r <= 1 for r in self.ratios):
            raise ValueError("All ratio values must be between 0 and 1.")
        if self.ratios[1] == 0:
            raise ValueError("Error: validation set ratio (val_ratio) cannot be zero. Early stopping requires a non-empty validation set.")


    def _create_atom_features(self, df_subset: pd.DataFrame):
        """Create atom-level descriptor features to describe which terminal is tail block.
        
        The D-MPNN model only considers non-hydrogen atoms when generating graph representations.
        The terminal atoms (heavy atoms closest to H2O in DFT calculations) and their neighbors
        are assigned a value of 1, second nearest neighbors 0.5, and all others 0.
        """
        atom_feature_weight = {'tail_atom_ids_r1': 1, 'tail_atom_ids_r2': 0.5}

        atom_features = []
        for mol_id in df_subset.index:
            mol = utils.make_mol(df_subset.loc[mol_id, self.smiles_label], keep_h=False, add_h=False)
            n_atoms = mol.GetNumAtoms()
            feature = np.zeros((n_atoms, 1))
            for rn_column, weight in atom_feature_weight.items():
                atom_ids_str = df_subset.loc[mol_id, rn_column]
                if pd.isnull(atom_ids_str) or not isinstance(atom_ids_str, str) or not atom_ids_str:
                    continue
                atom_ids = eval(atom_ids_str)
                if atom_ids:
                    feature[atom_ids, 0] = weight 
            atom_features.append(feature)
        
        return atom_features

    def _prepare_data(self):
        """Split data into train, validation, and test sets with cross-validation splits. Test set is a held-out set that will be only used for final evaluation."""
        train_ratio, val_ratio, test_ratio = self.ratios

        if test_ratio > 0:
            splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=self.random_state)
            original_indices = self.df_input.index
            temp_indices = np.arange(len(self.df_input))
            class_labels_for_split = self.df_input[self.class_label].values

            try:
                train_val_iloc_indices, test_iloc_indices = next(splitter_test.split(temp_indices, class_labels_for_split))
            except ValueError as e:
                print(f"Error during test set stratification: {e}")
                print("This might happen if a class is present in too few samples to be split according to the test_ratio.")
                print("Consider adjusting ratios, ensuring sufficient samples per class, or using a different splitting strategy if stratification is not strictly needed for the test set.")
                raise

            self.test_mol_ids = original_indices[test_iloc_indices].tolist()
            train_val_df = self.df_input.iloc[train_val_iloc_indices]
        else:
            self.test_mol_ids = []
            train_val_df = self.df_input.copy()

        if val_ratio > 0 and train_ratio > 0:
            val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)
            splitter_cv = StratifiedShuffleSplit(n_splits=self.n_splits_cv, test_size=val_ratio_in_train_val, random_state=self.random_state)

            original_train_val_indices = train_val_df.index
            temp_train_val_indices = np.arange(len(train_val_df))
            class_labels_for_cv_split = train_val_df[self.class_label].values

            for train_iloc_idx, val_iloc_idx in splitter_cv.split(temp_train_val_indices, class_labels_for_cv_split):
                train_ids = original_train_val_indices[train_iloc_idx].tolist()
                val_ids = original_train_val_indices[val_iloc_idx].tolist()
                if not val_ids:
                    raise ValueError("Validation set is required for following model training process as early stopping is implemented.")
                self.cv_splits.append((train_ids, val_ids))
        else:
            raise ValueError("Training ratio must be greater than 0.")


    def get_data_for_fold(self, fold_index: int):
        """Get data for a specific cross-validation fold."""
        if not (0 <= fold_index < len(self.cv_splits)):
            raise IndexError(f"Fold index {fold_index} is out of bounds for {len(self.cv_splits)} CV splits.")

        train_ids, val_ids = self.cv_splits[fold_index]

        df_train = self.df_input.loc[train_ids]
        df_val = self.df_input.loc[val_ids] if val_ids else pd.DataFrame()
        train_smis = df_train[self.smiles_label].tolist()
        train_targets = df_train[[self.target_label]].values
        train_mol = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in train_smis]
        train_atom_features_extra = self._create_atom_features(df_train)

        train_dset = data.MoleculeDataset([
            data.MoleculeDatapoint(
                train_mol[i],
                train_targets[i],
                V_d=train_atom_features_extra[i] if train_atom_features_extra else None,
            )
            for i in range(len(train_smis))], self.featurizer)

        val_smis = df_val[self.smiles_label].tolist()
        val_targets = df_val[[self.target_label]].values
        val_mol = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in val_smis]
        val_atom_features_extra = self._create_atom_features(df_val)
        val_dset = data.MoleculeDataset([
            data.MoleculeDatapoint(
                val_mol[i],
                val_targets[i],
                x_d=None,
                V_d=val_atom_features_extra[i] if val_atom_features_extra else None,
            )
            for i in range(len(val_smis))
        ], self.featurizer)

        return train_dset, val_dset

    def get_test_data(self):
        """Get the held-out test set data.

        Returns:
            tuple: (test_dset, test_targets) where test_targets are original unscaled labels.
                   Returns None if no test set exists.
        """
        if not self.test_mol_ids:
            return None

        df_test = self.df_input.loc[self.test_mol_ids]
        test_smis = df_test[self.smiles_label].tolist()
        test_targets = df_test[[self.target_label]].values
        test_mol = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in test_smis]
        test_atom_features_extra = self._create_atom_features(df_test)
        test_dset = data.MoleculeDataset([
            data.MoleculeDatapoint(
                test_mol[i],
                test_targets[i],
                V_d=test_atom_features_extra[i] if test_atom_features_extra else None,
            )
            for i in range(len(test_smis))
        ], self.featurizer)
        return test_dset, test_targets


class MPNNTrainer:
    """Manages training of a single MPNN model."""
    
    DEFAULT_CONFIG = {
        "batch_size": 16,
        "depth": 4,
        "ffn_hidden_dim": 300,
        "ffn_num_layers": 3,
        "message_hidden_dim": 300,
        "dropout": 0.2,
        "init_lr_ratio": 0.5, # init_lr = max_lr * init_lr_ratio
        "final_init_lr_ratio": 1e-3, # final_lr = max_lr * init_lr_ratio * final_init_lr_ratio
        "max_lr": 1e-3,
    }

    def __init__(self, model_config: dict = None,
                 metrics=None,
                 max_epochs: int = 50,
                 num_workers: int = 1,
                 patience_early_stopping: int = 10,
                 accelerator: str = "auto",
                 checkpoint_dir: str = "./mpnn_checkpoints",
                 enable_progress_bar: bool = True):

        self.config = {**self.DEFAULT_CONFIG, **(model_config or {})}
        self.metrics = metrics if metrics is not None else [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric(), nn.metrics.R2Metric()]
        self.max_epochs = max_epochs
        self.num_workers = num_workers
        self.patience_early_stopping = patience_early_stopping
        self.accelerator = accelerator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enable_progress_bar = enable_progress_bar
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.model = None
        self.trainer = None

    def _create_model(self, output_transform=None):
        """Initialize MPNN model with extra atom descriptors dimension of 1."""
        d_vd = 1

        mp = nn.BondMessagePassing(
            d_v=self.featurizer.atom_fdim,
            d_vd=d_vd,
            d_h=self.config["message_hidden_dim"],
            depth=self.config["depth"],
            dropout=self.config["dropout"]
        )
        # AttentiveAggregation will improve the performance of the model
        agg = nn.AttentiveAggregation(output_size=mp.output_dim)
        ffn_input_dim = mp.output_dim

        ffn = nn.RegressionFFN(
            input_dim=ffn_input_dim,
            output_transform=output_transform,
            hidden_dim=self.config["ffn_hidden_dim"],
            n_layers=self.config["ffn_num_layers"],
            dropout=self.config["dropout"],
            activation='leakyReLU'
        )

        init_lr = self.config["init_lr_ratio"] * self.config["max_lr"]
        final_lr = self.config["final_init_lr_ratio"] * init_lr

        model = models.MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            batch_norm=False,
            metrics=self.metrics,
            init_lr=init_lr,
            max_lr=self.config["max_lr"],
            final_lr=final_lr,
        )
        return model

    def train(self, train_dset: data.MoleculeDataset, val_dset: data.MoleculeDataset):
        """Train MPNN model.
        
        Args:
            train_dset: Training dataset
            val_dset: Validation dataset (required for early stopping)
        """
        targets_scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(targets_scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(targets_scaler)

        self.model = self._create_model(output_transform=output_transform)

        train_loader = data.build_dataloader(
            train_dset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=True
        )
        val_loader = data.build_dataloader(
            val_dset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=False
        )

        callbacks = []
        monitor_metric = 'val_loss'
        monitor_mode = "min" if self.metrics[0].minimize else "max"

        if self.patience_early_stopping > 0:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                patience=self.patience_early_stopping,
                verbose=False,
                mode=monitor_mode
            )
            callbacks.append(early_stop_callback)

        model_checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='best_model-{epoch:02d}-{' + monitor_metric + ':.4f}',
            save_top_k=1,
            monitor=monitor_metric,
            mode=monitor_mode,
            verbose=False,
            save_last=False,
            auto_insert_metric_name=False
        )
        callbacks.append(model_checkpoint_callback)

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            enable_checkpointing=True,
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=self.enable_progress_bar,
            logger=False,
            check_val_every_n_epoch=1,
            enable_model_summary=False,
            reload_dataloaders_every_n_epochs=0,
            num_sanity_val_steps=0,
            inference_mode=False,
        )

        self.trainer.fit(self.model, train_loader, val_loader)

        best_model_path = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and hasattr(callback, 'best_model_path'):
                best_model_path = callback.best_model_path
                break

        if best_model_path and Path(best_model_path).exists():
            if self.enable_progress_bar:
                print(f"Best model saved at: {best_model_path}")
            try:
                self.model = models.MPNN.load_from_checkpoint(best_model_path)
            except Exception as e:
                print(f"Warning: Failed to load best model: {e}")

        return self.trainer.callback_metrics.get(monitor_metric, None)

    def predict(self, data_to_predict: data.MoleculeDataset):
        """Make predictions using the trained model.
        
        Args:
            data_to_predict: Dataset to predict on
            
        Returns:
            numpy.ndarray: Prediction results
        """
        if not self.model:
            raise RuntimeError("Model has not been trained or loaded.")

        loader = data.build_dataloader(
            data_to_predict,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=False
        )
        trainer_instance = self.trainer if self.trainer else pl.Trainer(
            accelerator=self.accelerator, 
            logger=False, 
            enable_checkpointing=False, 
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
        )
        raw_predictions = trainer_instance.predict(model=self.model, dataloaders=loader)
        predictions = torch.cat([p.squeeze() for p in raw_predictions]).cpu().numpy()

        return predictions.flatten()

    def evaluate(self, test_data: data.MoleculeDataset, test_targets):
        """Evaluate model on test data.
        
        Args:
            test_data: Test dataset
            test_targets: Test target values (can be DataFrame, numpy array or list)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        predictions = self.predict(test_data)
        
        if hasattr(test_targets, 'values'):
            y_true = test_targets.values.flatten()
        elif hasattr(test_targets, 'flatten'):
            y_true = test_targets.flatten()
        else:
            y_true = np.array(test_targets).flatten()

        rmse = root_mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R2: {r2:.4f}")
        return {"RMSE": rmse, "MAE": mae, "R2": r2, "predictions": predictions, "true_values": y_true}

    def save_model(self, file_path: str):
        """Save model to specified path.
        
        Args:
            file_path: Save path
        """
        if not self.model:
            raise RuntimeError("No model to save. Please train the model first.")
        
        save_path = Path(file_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.trainer.save_checkpoint(str(save_path))
        print(f"Model saved to: {save_path}")

    def load_model(self, file_path: str):
        """Load model from specified path.
        
        Args:
            file_path: Model file path
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            self.model = models.MPNN.load_from_checkpoint(file_path)
            print(f"Model loaded from: {file_path}")
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {file_path}: {e}")


class MPNNHyperparameterOptimizer:
    """Manages hyperparameter optimization using Ray Tune."""
    
    DEFAULT_SEARCH_SPACE = {
        "batch_size": tune.choice([16, 32, 64]), 
        "depth": tune.choice([3, 4, 5]),
        "ffn_hidden_dim": tune.choice([200, 300, 400]),
        "ffn_num_layers": tune.choice([2, 3, 4]),
        "message_hidden_dim": tune.choice([200, 300, 400]),
        "dropout": tune.choice([0.1, 0.2, 0.4]),
        "init_lr_ratio": tune.choice([0.01, 0.05, 0.1, 0.5]),
        "final_init_lr_ratio": tune.choice([1e-3, 1e-2, 1e-1]),
        "max_lr": tune.choice([5e-4, 1e-3, 1e-2]),
    }

    def __init__(self, data_manager: DataManager,
                 search_space: dict = None,
                 chemprop_metrics = None,
                 num_samples_tune: int = 20,
                 max_epochs_per_trial: int = 50,
                 patience_per_trial: int = 10,
                 num_workers_per_trial: int = 1,
                 cpus_per_trial: int = 5,
                 gpus_per_trial: float = 0.5,
                 results_dir: str = "./ray_tune_results",
                 experiment_name: str = "mpnn_hpo"):
        """Initialize hyperparameter optimizer.

        Args:
            data_manager (DataManager): DataManager instance.
            search_space (Optional[Dict[str, Any]]): Ray Tune search space dictionary.
                If None, uses default DEFAULT_SEARCH_SPACE.
            chemprop_metrics (Optional[List[nn.metrics.Metric]]): Chemprop metric objects list
                (e.g. nn.metrics.RMSEMetric()) for evaluation on each fold.
                These are only for reporting; optimization target is always avg_val_loss.
                Defaults to [RMSEMetric, MAEMetric, R2Metric].
            num_samples_tune (int): Number of hyperparameter combinations to try.
            max_epochs_per_trial (int): Maximum training epochs per trial.
            patience_per_trial (int): Early stopping patience per trial.
            num_workers_per_trial (int): DataLoader worker processes per trial.
            cpus_per_trial (int): CPU cores allocated per trial.
            gpus_per_trial (int): GPU count allocated per trial.
            results_dir (str): Directory to save Ray Tune results.
            experiment_name (str): Name of the Ray Tune experiment.
        """
        configure_silent_mode()
        
        self.data_manager = data_manager
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE.copy()

        if chemprop_metrics is None:
            self.chemprop_metrics = [
                nn.metrics.RMSEMetric(), 
                nn.metrics.MAEMetric(), 
                nn.metrics.R2Metric()
            ]
        else:
            self.chemprop_metrics = chemprop_metrics
        self.monitor_mode = "min" if self.chemprop_metrics[0].minimize else "max"
        self.num_samples_tune = num_samples_tune
        self.max_epochs_per_trial = max_epochs_per_trial
        self.patience_per_trial = patience_per_trial
        self.num_workers_per_trial = num_workers_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.results_dir = Path(results_dir).resolve()
        self.experiment_name = experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.best_config = None
        self.analysis = None

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=False,
                logging_level=logging.ERROR,
            )


    @staticmethod
    def _trainable_function(config, data_manager_ref, chemprop_metrics, max_epochs_per_trial, patience_per_trial, num_workers_per_trial, gpus_per_trial):
        """Ray Tune trainable function executed for each trial.

        Uses cross-validation: trains on all folds for each hyperparameter set and takes
        the average validation loss.

        Args:
            config (Dict[str, Any]): Hyperparameter dictionary provided by Ray Tune.
            data_manager_ref (ray.ObjectRef): Ray object reference to DataManager instance.
            chemprop_metrics: Chemprop metrics list.
            max_epochs_per_trial: Maximum epochs per trial.
            patience_per_trial: Early stopping patience per trial.
            num_workers_per_trial: DataLoader worker processes per trial.
            gpus_per_trial: GPU count per trial.
        """
        configure_silent_mode()
        
        dm: DataManager = data_manager_ref
        default_val_loss = float('inf') if chemprop_metrics[0].minimize else float('-inf')

        if dm.n_splits_cv == 0:
            print("Warning: DataManager has 0 CV splits. HPO cannot proceed without train/val data.")
            train.report({"avg_val_loss": default_val_loss})
            return

        fold_training_results = [] 
        
        for fold_idx in range(dm.n_splits_cv):
           
            train_data, val_data = dm.get_data_for_fold(fold_idx)

            if val_data is None or len(val_data) == 0:
                print(f"Warning: No validation data available for fold {fold_idx}. Skipping this fold.")
                continue

            trial_dir = train.get_context().get_trial_dir()
            fold_checkpoint_dir = str(Path(trial_dir) / "checkpoints" / f"fold_{fold_idx}")
            
            trainer = MPNNTrainer(
                model_config=config,
                metrics=chemprop_metrics,
                max_epochs=max_epochs_per_trial,
                patience_early_stopping=patience_per_trial,
                num_workers=num_workers_per_trial,
                accelerator="cpu" if gpus_per_trial == 0 else "gpu",
                checkpoint_dir=fold_checkpoint_dir,
                enable_progress_bar=False 
            )

            trainer.train(train_data, val_data)
            lightning_metrics = trainer.trainer.callback_metrics
            fold_val_loss_tensor = lightning_metrics.get('val_loss')
            if fold_val_loss_tensor is not None:
                val_loss = fold_val_loss_tensor.item() if isinstance(fold_val_loss_tensor, torch.Tensor) else float(fold_val_loss_tensor)
                if not np.isinf(val_loss):
                    fold_training_results.append(val_loss)
                else:
                    train.report({"avg_val_loss": default_val_loss})
                    return
            else:
                train.report({"avg_val_loss": default_val_loss})
                return
        val_losses = fold_training_results
        avg_val_loss = np.mean(val_losses)       
        train.report({"avg_val_loss": avg_val_loss})

    def run_optimization(self):
        """Execute hyperparameter optimization process.

        Returns:
            tuple: Tuple containing:
                - best_config (Optional[Dict[str, Any]]): Best hyperparameter configuration found.
                  None if optimization failed or no results.
                - analysis (Optional[tune.ExperimentAnalysis]): Ray Tune ExperimentAnalysis object.
        """
        scheduler = ASHAScheduler(
            max_t=self.max_epochs_per_trial,
            grace_period=20, 
            reduction_factor=4,
            brackets=1
        )

        optuna_search = OptunaSearch()
        data_manager_obj = self.data_manager

        trainable_with_resources = tune.with_resources(
            MPNNHyperparameterOptimizer._trainable_function,
            {"cpu": self.cpus_per_trial, "gpu": self.gpus_per_trial}
        )

        tuner = tune.Tuner(
            tune.with_parameters(
                trainable_with_resources, 
                data_manager_ref=data_manager_obj, 
                chemprop_metrics=self.chemprop_metrics, 
                max_epochs_per_trial=self.max_epochs_per_trial, 
                patience_per_trial=self.patience_per_trial, 
                num_workers_per_trial=self.num_workers_per_trial, 
                gpus_per_trial=self.gpus_per_trial
            ),
            param_space=self.search_space,
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                scheduler=scheduler,
                num_samples=self.num_samples_tune,
                metric="avg_val_loss",
                mode=self.monitor_mode,
            ),
            run_config=train.RunConfig(
                name=self.experiment_name,
                storage_path=str(self.results_dir),
                stop={"training_iteration": self.max_epochs_per_trial}, 
                failure_config=train.FailureConfig(max_failures=-1), 
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=1, 
                    checkpoint_score_attribute="avg_val_loss",
                    checkpoint_score_order=self.monitor_mode
                )
            ),
        )

        self.analysis = tuner.fit()
        
        best_result = self.analysis.get_best_result(
            metric="avg_val_loss", 
            mode=self.monitor_mode
        )

        if best_result and best_result.config:
            self.best_config = best_result.config
            print("Hyperparameter optimization finished.")
            print(f"Best hyperparameters found for avg_val_loss (min):")
            for param, value in self.best_config.items():
                print(f"  {param}: {value}")
        else:
            self.best_config = None
            print("Hyperparameter optimization finished, but no best configuration was found or an error occurred.")
            if self.analysis:
                print(f"Number of trials: {len(self.analysis.trials)}")
                for i, trial in enumerate(self.analysis.trials):
                    if hasattr(trial, 'status') and trial.status == "ERROR":
                        print(f"Trial {trial.trial_id} (index {i}) had an error.")

        return self.best_config, self.analysis

    def get_best_model_config(self):
        """Get the best model configuration.
        
        Returns:
            dict: Best hyperparameter configuration, None if optimization hasn't been run
        """
        if self.best_config:
            return self.best_config
        else:
            print("Optimization has not been run or no best configuration was found.")
            return None


if __name__ == '__main__':
    print("Setting up example data...")

    df_example = pd.read_csv('data/Eint_H2O_data.csv', index_col=0)

    TARGET_COLUMN = 'Eint_H2O'
    SMILES_COLUMN = 'CanonSMILES'
    CLASS_COLUMN = 'Class_label'

    print("\n--- Initializing DataManager ---")
    data_manager = DataManager(
        df_input=df_example,
        target_label=TARGET_COLUMN,
        smiles_label=SMILES_COLUMN,
        class_label=CLASS_COLUMN,
        ratios=(0.8, 0.1, 0.1),
        n_splits_cv=10
    )
    print(f"Number of CV splits configured: {data_manager.n_splits_cv}")
    print(f"Test molecule IDs ({len(data_manager.test_mol_ids)}): {data_manager.test_mol_ids[:5]}...")
    if data_manager.cv_splits:
        train_ids_ex, val_ids_ex = data_manager.cv_splits[0]
        print(f"First CV fold - Train IDs ({len(train_ids_ex)}): {train_ids_ex[:5]}...")
        print(f"First CV fold - Val IDs ({len(val_ids_ex)}): {val_ids_ex[:5] if val_ids_ex else 'None'}...")

    print("\n--- Single Model Training ---")
    if data_manager.cv_splits:
        train_data_single, val_data_single = data_manager.get_data_for_fold(0)

        mpnn_trainer = MPNNTrainer(
            model_config={"max_lr": 0.0005, "depth": 2, "dropout": 0.1},
            max_epochs=10,
            patience_early_stopping=3,
            enable_progress_bar=True
        )
        mpnn_trainer.train(train_data_single, val_data_single)
        mpnn_trainer.save_model("./single_mpnn_model.ckpt")

        test_result = data_manager.get_test_data()
        if test_result:
            test_data_single, test_targets_single_original = test_result
            print("\n--- Evaluating Single Model on Test Set ---")
            eval_results_single = mpnn_trainer.evaluate(test_data_single, test_targets_single_original)
        else:
            print("No test data to evaluate single model.")
    else:
        print("No CV splits available in DataManager to train a single model.")

