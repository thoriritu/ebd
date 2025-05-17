import threading
import json
from datetime import datetime, timedelta
import os
import joblib
import pickle
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kafka import KafkaConsumer
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi import HTTPException
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from fastapi.encoders import jsonable_encoder
from scipy.spatial import ConvexHull
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score, silhouette_samples
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.manifold import TSNE

# Constants
MODEL_DIR = 'stored-models/performance-energy'
os.makedirs(MODEL_DIR, exist_ok=True)

# Constants
MODEL_DIR = 'stored-models/performance-energy'
VISUALIZATION_DIR = os.path.join(MODEL_DIR, 'visualizations')  # Add this line
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)  # Add this line

# Model paths
PATHS = {
    'performance': os.path.join(MODEL_DIR, 'performance_clf.pkl'),
    'energy': os.path.join(MODEL_DIR, 'energy_clf.pkl'),
    'scaler': os.path.join(MODEL_DIR, 'scaler.pkl'),
    'affinity': os.path.join(MODEL_DIR, 'affinity.pkl'),
    'state': os.path.join(MODEL_DIR, 'state.pkl'),
    'training_data': os.path.join(MODEL_DIR, 'training_data.pkl'),
    'energy_forecaster': os.path.join(MODEL_DIR, 'energy_forecaster.pkl'),
    'performance_forecaster': os.path.join(MODEL_DIR, 'performance_forecaster.pkl'),
    'anomaly_detector': os.path.join(MODEL_DIR, 'anomaly_detector.pkl'),
    'time_encoder': os.path.join(MODEL_DIR, 'time_encoder.pkl'),
    'performance_cluster': os.path.join(MODEL_DIR, 'performance_cluster.pkl'),
    'energy_cluster': os.path.join(MODEL_DIR, 'energy_cluster.pkl'),
    'performance_scaler': os.path.join(MODEL_DIR, 'performance_scaler.pkl'),
    'energy_scaler': os.path.join(MODEL_DIR, 'energy_scaler.pkl')
}

# Feature definitions
FEATURES = {
    'performance': ['Time_per_Unit', 'Defect_Rate', 'Affinity_Score'],
    'energy': ['Energy_per_Unit', 'Power_Consumption_kW', 'Temperature_C']
}


class ModelValidation:
    @staticmethod
    def evaluate_clusters(X, labels):
        """Evaluate clustering performance"""
        if len(set(labels)) > 1:  # Need at least 2 clusters for these metrics
            return {
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels)
            }
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "warning": "Not enough clusters for meaningful evaluation"
        }

    @staticmethod
    def evaluate_forecasts(y_true, y_pred):
        """Evaluate regression forecasts"""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "error_distribution": {
                "mean": float(np.mean(y_pred - y_true)),
                "std": float(np.std(y_pred - y_true)),
                "min": float(np.min(y_pred - y_true)),
                "max": float(np.max(y_pred - y_true))
            }
        }

    @staticmethod
    def time_series_cv(model, X, y, n_splits=5):
        """Time-series cross validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics.append(ModelValidation.evaluate_forecasts(y_test, y_pred))

        return metrics

class ClusterModels:
    def __init__(self, model_manager):
        self.performance_cluster = None
        self.energy_cluster = None
        self.performance_scaler = StandardScaler()
        self.energy_scaler = StandardScaler()
        self.time_encoder = LabelEncoder()
        self.state = {}
        self.initialize_models()
        self.model_manager = model_manager

    def initialize_models(self):
        """Initialize models using historical data"""
        # Load and merge historical data
        print("Loading historical data for model initialization...")
        try:
            iot = pd.read_csv('data/generated_data/historical-iot.csv', parse_dates=['Timestamp'])
            scada = pd.read_csv('data/generated_data/historical-scada.csv', parse_dates=['Timestamp'])
            mes = pd.read_csv('data/generated_data/historical-mes.csv', parse_dates=['Timestamp'])

            # Merge datasets
            data = (
                mes.set_index(['Timestamp', 'Machine_ID'])
                .join(scada.set_index(['Timestamp', 'Machine_ID']), how='left')
                .join(iot.set_index(['Timestamp', 'Machine_ID']), how='left')
                .reset_index()
            )
            data.ffill(inplace=True)

            # Calculate derived metrics
            data['Time_per_Unit'] = data['Production_Time_min'] / data['Units_Produced']
            data['Defect_Rate'] = data['Defective_Units'] / data['Units_Produced']
            data['Energy_per_Unit'] = data['Power_Consumption_kW'] / data['Units_Produced']

            # Initialize performance cluster model
            if os.path.exists(PATHS['performance_cluster']):
                self.performance_cluster = joblib.load(PATHS['performance_cluster'])
            else:
                print("Initializing performance cluster model with historical data...")
                perf_features = data[['Time_per_Unit', 'Defect_Rate']].values
                self.performance_cluster = KMeans(n_clusters=5, random_state=42)
                self.performance_cluster.fit(perf_features)
                joblib.dump(self.performance_cluster, PATHS['performance_cluster'])

            # Initialize energy cluster model
            if os.path.exists(PATHS['energy_cluster']):
                self.energy_cluster = joblib.load(PATHS['energy_cluster'])
            else:
                print("Initializing energy cluster model with historical data...")
                energy_features = data[['Power_Consumption_kW', 'Temperature_C', 'Vibration_mm_s']].values
                self.energy_cluster = KMeans(n_clusters=5, random_state=42)
                self.energy_cluster.fit(energy_features)
                joblib.dump(self.energy_cluster, PATHS['energy_cluster'])

            # Initialize scalers
            if os.path.exists(PATHS['performance_scaler']):
                self.performance_scaler = joblib.load(PATHS['performance_scaler'])
            else:
                self.performance_scaler.fit(data[['Time_per_Unit', 'Defect_Rate']].values)
                joblib.dump(self.performance_scaler, PATHS['performance_scaler'])

            if os.path.exists(PATHS['energy_scaler']):
                self.energy_scaler = joblib.load(PATHS['energy_scaler'])
            else:
                self.energy_scaler.fit(data[['Power_Consumption_kW', 'Temperature_C', 'Vibration_mm_s']].values)
                joblib.dump(self.energy_scaler, PATHS['energy_scaler'])

            # Initialize affinity scores
            if 'affinity_scores' not in self.state:
                print("Calculating initial affinity scores...")
                self.state['affinity_scores'] = {}
                for _, row in data.iterrows():
                    key = (row['Operator_ID'], row['Machine_ID'])
                    success_rate = 1 - (row['Defective_Units'] / row['Units_Produced'])
                    self.state['affinity_scores'][key] = success_rate

            print("Model initialization complete using historical data")

        except Exception as e:
            print(f"Error initializing models with historical data: {str(e)}")
            raise

    def prepare_performance_features(self, machine_data, operator_id, timestamps):
        """Prepare performance features for clustering"""
        if not isinstance(machine_data, dict) or 'machine_id' not in machine_data:
            return None

        try:
            operator_id = int(operator_id)
        except (ValueError, TypeError):
            operator_id = 0

        affinity = self.model_manager.affinity_scores.get((operator_id, machine_data['machine_id']), 0.5)
        features = []
        for ts in timestamps:
            features.append([
                pd.to_datetime(ts).hour,
                machine_data.get('Defect_Rate', 0),
                affinity
            ])
        return np.array(features) if features else None

    def prepare_energy_features(self, machine_data, timestamps):
        """Prepare energy features for clustering"""
        features = []
        for ts in timestamps:
            features.append([
                pd.to_datetime(ts).hour,
                machine_data.get('Temperature_C', 25),
                machine_data.get('Vibration_mm_s', 0)
            ])
        return np.array(features) if features else None

    def update_cluster_models(self):
        """Update clustering models with latest data"""
        try:
            # Update performance clusters
            perf_features = []
            for machine_id, data in self.state.items():
                if isinstance(data, dict) and 'history' in data and 'Operator_ID' in data:
                    operator_id = data['Operator_ID']
                    features = self.prepare_performance_features(
                        data, operator_id,
                        [rec['timestamp'] for rec in data['history'][-24:] if 'timestamp' in rec]
                    )
                    if features is not None:
                        perf_features.extend(features)

            if perf_features:
                X_perf = np.array(perf_features)
                X_perf_scaled = self.performance_scaler.fit_transform(X_perf)
                self.performance_cluster.fit(X_perf_scaled)
                joblib.dump(self.performance_cluster, PATHS['performance_cluster'])
                joblib.dump(self.performance_scaler, PATHS['performance_scaler'])

            # Update energy clusters
            energy_features = []
            for machine_id, data in self.state.items():
                if isinstance(data, dict) and 'history' in data:
                    features = self.prepare_energy_features(
                        data,
                        [rec['timestamp'] for rec in data['history'][-24:] if 'timestamp' in rec]
                    )
                    if features is not None:
                        energy_features.extend(features)

            if energy_features:
                X_energy = np.array(energy_features)
                X_energy_scaled = self.energy_scaler.fit_transform(X_energy)
                self.energy_cluster.fit(X_energy_scaled)
                joblib.dump(self.energy_cluster, PATHS['energy_cluster'])
                joblib.dump(self.energy_scaler, PATHS['energy_scaler'])

        except Exception as e:
            print(f"Error updating cluster models: {str(e)}")
            raise

    def persist_models(self):
        """Persist all models and state"""
        joblib.dump(self.performance_cluster, PATHS['performance_cluster'])
        joblib.dump(self.energy_cluster, PATHS['energy_cluster'])
        joblib.dump(self.performance_scaler, PATHS['performance_scaler'])
        joblib.dump(self.energy_scaler, PATHS['energy_scaler'])
        joblib.dump(self.time_encoder, PATHS['time_encoder'])
        with open(PATHS['state'], 'wb') as f:
            pickle.dump(self.state, f)

class ModelManager:
    def __init__(self):
        self.performance_clf = None
        self.energy_clf = None
        self.scaler = None
        self.affinity_scores = {}
        self.state = {}
        self.training_data = {}
        self.validator = ModelValidation()
        self.validation_metrics = {
            'performance_cluster': {'training': None, 'current': None},
            'energy_cluster': {'training': None, 'current': None}
        }
        self.initialize_models()

    def initialize_models(self):
        """Initialize or load all models and data"""
        print("Loading historical data and initializing models...")

        # Load or create affinity scores
        if os.path.exists(PATHS['affinity']):
            self.affinity_scores = joblib.load(PATHS['affinity'])
        else:
            self._calculate_affinity_scores()

        # Load or create scaler
        if os.path.exists(PATHS['scaler']):
            self.scaler = joblib.load(PATHS['scaler'])
        else:
            self.scaler = StandardScaler()
            self._fit_scaler()

        # Initialize validation metrics structure
        self.validation_metrics = {
            'performance_cluster': {'training': None, 'current': None},
            'energy_cluster': {'training': None, 'current': None}
        }

        # Load or create performance model
        if os.path.exists(PATHS['performance']):
            try:
                model_data = joblib.load(PATHS['performance'])
                # Handle both old (model only) and new (model + metrics) formats
                if isinstance(model_data, dict):
                    self.performance_clf = model_data['model']
                    self.validation_metrics['performance_cluster']['training'] = model_data.get('training_metrics')
                else:
                    self.performance_clf = model_data
                    # For old format, we'll need to calculate training metrics
                    print("Legacy performance model format detected - recalculating training metrics")
                    self._train_performance_model()  # This will save in new format
            except Exception as e:
                print(f"Error loading performance model: {str(e)} - retraining")
                self._train_performance_model()
        else:
            self._train_performance_model()

        # Load or create energy model
        if os.path.exists(PATHS['energy']):
            try:
                model_data = joblib.load(PATHS['energy'])
                # Handle both old (model only) and new (model + metrics) formats
                if isinstance(model_data, dict):
                    self.energy_clf = model_data['model']
                    self.validation_metrics['energy_cluster']['training'] = model_data.get('training_metrics')
                else:
                    self.energy_clf = model_data
                    # For old format, we'll need to calculate training metrics
                    print("Legacy energy model format detected - recalculating training metrics")
                    self._train_energy_model()  # This will save in new format
            except Exception as e:
                print(f"Error loading energy model: {str(e)} - retraining")
                self._train_energy_model()
        else:
            self._train_energy_model()

        # Load or initialize state
        if os.path.exists(PATHS['state']):
            with open(PATHS['state'], 'rb') as f:
                self.state = pickle.load(f)
            print(f"Loaded state for {len(self.state)} machines.")
        else:
            self.state = {}
            print("Initialized empty state.")

    def _calculate_affinity_scores(self):
        """Calculate operator-machine affinity scores"""
        data = self._load_and_merge_data()
        success_rates = data.groupby(['Operator_ID', 'Machine_ID']).apply(
            lambda x: (1 - (x['Defective_Units'].sum() / x['Units_Produced'].sum()))
        ).reset_index(name='Success_Rate')
        self.affinity_scores = success_rates.set_index(['Operator_ID', 'Machine_ID'])['Success_Rate'].to_dict()
        joblib.dump(self.affinity_scores, PATHS['affinity'])

    def _load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge all historical data"""
        mes = pd.read_csv('data/generated_data/historical-mes.csv', parse_dates=['Timestamp'])
        scada = pd.read_csv('data/generated_data/historical-scada.csv', parse_dates=['Timestamp'])
        iot = pd.read_csv('data/generated_data/historical-iot.csv', parse_dates=['Timestamp'])

        data = (
            mes.set_index(['Timestamp', 'Machine_ID'])
            .join(scada.set_index(['Timestamp', 'Machine_ID']), how='left')
            .join(iot.set_index(['Timestamp', 'Machine_ID']), how='left')
            .reset_index()
        )
        data.ffill(inplace=True)

        # Calculate metrics
        data['Time_per_Unit'] = data['Production_Time_min'] / data['Units_Produced']
        data['Defect_Rate'] = (data['Defective_Units'] / data['Units_Produced']) * 100
        data['Energy_per_Unit'] = data['Power_Consumption_kW'] / data['Units_Produced']
        data['Affinity_Score'] = data.apply(
            lambda row: self.affinity_scores.get((row['Operator_ID'], row['Machine_ID']), 0.5), axis=1
        )

        return data

    def _fit_scaler(self):
        """Fit the scaler on energy features"""
        data = self._load_and_merge_data()
        self.scaler.fit(data[FEATURES['energy']].values)
        joblib.dump(self.scaler, PATHS['scaler'])


    def _train_performance_model(self):
        """Train the performance clustering model with validation"""
        data = self._load_and_merge_data()
        X = data[FEATURES['performance']].values

        # Train model
        self.performance_clf = KMeans(n_clusters=3, random_state=42)
        labels = self.performance_clf.fit_predict(X)

        # Evaluate clustering
        training_metrics = {
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "cluster_distribution": dict(zip(*np.unique(labels, return_counts=True)))
        }

        # Store both model and metrics
        model_data = {
            'model': self.performance_clf,
            'training_metrics': training_metrics
        }
        joblib.dump(model_data, PATHS['performance'])

        # Generate visualization
        self.plot_performance_clusters()
        self.plot_performance_clusters_enhanced()
        self.plot_enhanced_performance_clusters()

        # Update validation metrics
        self.validation_metrics['performance_cluster'] = {
            'training': training_metrics,
            'current': training_metrics  # Initially current = training
        }

        return self.performance_clf

    # def _train_energy_model(self):
    #     """Train the energy clustering model with validation"""
    #     data = self._load_and_merge_data()
    #     X = self.scaler.transform(data[FEATURES['energy']].values)
    #
    #     # Train model
    #     self.energy_clf = DBSCAN(eps=0.5, min_samples=5)
    #     labels = self.energy_clf.fit_predict(X)
    #
    #     # Evaluate clustering
    #     training_metrics = {
    #         "silhouette": silhouette_score(X, labels) if len(set(labels)) > 1 else None,
    #         "calinski_harabasz": calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else None,
    #         "davies_bouldin": davies_bouldin_score(X, labels) if len(set(labels)) > 1 else None,
    #         "anomaly_percentage": np.mean(np.array(labels) == -1),
    #         "cluster_distribution": dict(zip(*np.unique(labels, return_counts=True)))
    #     }
    #
    #     # Store both model and metrics
    #     model_data = {
    #         'model': self.energy_clf,
    #         'training_metrics': training_metrics
    #     }
    #     joblib.dump(model_data, PATHS['energy'])
    #
    #     # Generate visualization
    #     self.plot_energy_clusters()
    #
    #     # Update validation metrics
    #     self.validation_metrics['energy_cluster'] = {
    #         'training': training_metrics,
    #         'current': training_metrics  # Initially current = training
    #     }
    #
    #     return self.energy_clf

    def _train_energy_model(self):
        """Train the energy clustering model"""
        data = self._load_and_merge_data()
        X_energy = self.scaler.transform(data[FEATURES['energy']].values)
        self.energy_clf = DBSCAN(eps=0.5, min_samples=5)
        # self.energy_clf.fit(X_energy)
        labels = self.energy_clf.fit_predict(X_energy)

        # self.validation_metrics['energy_cluster'] = {
        #     "training": self.validator.evaluate_clusters(X_energy, labels),
        #     "cluster_distribution": dict(zip(*np.unique(labels, return_counts=True))),
        #     "anomaly_percentage": np.mean(np.array(labels) == -1) if len(labels) > 0 else 0
        # }

        training_metrics = {
            "silhouette": silhouette_score(X_energy, labels) if len(set(labels)) > 1 else None,
            "calinski_harabasz": calinski_harabasz_score(X_energy, labels) if len(set(labels)) > 1 else None,
            "davies_bouldin": davies_bouldin_score(X_energy, labels) if len(set(labels)) > 1 else None,
            "anomaly_percentage": np.mean(np.array(labels) == -1),
            "cluster_distribution": dict(zip(*np.unique(labels, return_counts=True)))
        }

        # joblib.dump(self.energy_clf, PATHS['energy'])
        joblib.dump({
            'model': self.energy_clf,
            'training_metrics': training_metrics
        }, PATHS['energy'])

        # Generate visualization
        self.plot_energy_clusters()
        self.plot_energy_clusters_enhanced()

        self.validation_metrics['energy_cluster']['training'] = training_metrics
        self.validation_metrics['energy_cluster']['current'] = training_metrics  # Initial current = training

        return self.energy_clf

    def analyze_performance(self, machine_id: str) -> dict:
        """
        Comprehensive performance analysis including:
        - Tier classification
        - Benchmark comparisons
        - Operator recommendations
        - Improvement targets
        """
        # Validate machine exists
        if machine_id not in self.state:
            raise ValueError(f"Machine {machine_id} not found in state")

        # Validate required features exist
        required_features = FEATURES['performance']
        if not all(feat in self.state[machine_id] for feat in required_features):
            missing = [f for f in required_features if f not in self.state[machine_id]]
            raise ValueError(f"Missing features for {machine_id}: {missing}")

        try:
            # 1. Prepare features and predict cluster
            features = np.array([
                [
                    self.state[machine_id]['Time_per_Unit'],
                    self.state[machine_id]['Defect_Rate'],
                    self.state[machine_id]['Affinity_Score']
                ]
            ])
            cluster = int(self.performance_clf.predict(features)[0])
            centers = self.performance_clf.cluster_centers_

            # 2. Tier classification
            tier_map = {0: "high", 1: "average", 2: "low"}
            current_tier = tier_map[cluster]

            # 3. Get operator recommendations
            operators = list(self.state[machine_id].get('operators', []))
            top_operators = []
            for op in operators:
                affinity = self.affinity_scores.get((op, machine_id), 0.5)
                top_operators.append({
                    "operator_id": op,
                    "affinity_score": float(affinity),
                    "recommended": affinity >= 0.8
                })
            # Sort by highest affinity
            top_operators.sort(key=lambda x: x["affinity_score"], reverse=True)

            # 4. Calculate improvement targets
            improvement = {}
            if cluster > 0:  # Can improve to next tier
                target_tier = tier_map[cluster - 1]
                improvement = {
                    "target_tier": target_tier,
                    "required_time_improvement": float(centers[cluster][0] - centers[cluster - 1][0]),
                    "required_defect_improvement": float(centers[cluster][1] - centers[cluster - 1][1])
                }

            return {
                "machine_id": machine_id,
                "current_tier": current_tier,
                "metrics": {
                    "time_per_unit": float(self.state[machine_id]['Time_per_Unit']),
                    "defect_rate": float(self.state[machine_id]['Defect_Rate']),
                    "affinity_score": float(self.state[machine_id]['Affinity_Score'])
                },
                "cluster_center": [float(x) for x in centers[cluster].tolist()],
                "top_operators": top_operators[:3],  # Return top 3
                "improvement_targets": improvement
            }

        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")


    def validate_models(self):
        """Run comprehensive validation on all models"""
        try:
            data = self._load_and_merge_data()

            # Performance Cluster Validation
            if self.performance_clf is not None:
                X_perf = data[FEATURES['performance']].values
                perf_labels = self.performance_clf.predict(X_perf)
                current_metrics = self.validator.evaluate_clusters(X_perf, perf_labels)

                # Convert numpy types in cluster distribution
                if 'cluster_distribution' in current_metrics:
                    current_metrics['cluster_distribution'] = {
                        int(k): int(v) for k, v in current_metrics['cluster_distribution'].items()
                    }

                # If no training metrics exist, use current as training baseline
                if self.validation_metrics['performance_cluster']['training'] is None:
                    self.validation_metrics['performance_cluster']['training'] = current_metrics

                self.validation_metrics['performance_cluster']['current'] = current_metrics

            # Energy Cluster Validation
            if self.energy_clf is not None:
                X_energy = self.scaler.transform(data[FEATURES['energy']].values)
                energy_labels = self.energy_clf.fit_predict(X_energy)
                current_metrics = self.validator.evaluate_clusters(X_energy, energy_labels)

                # Convert numpy types in cluster distribution
                if 'cluster_distribution' in current_metrics:
                    current_metrics['cluster_distribution'] = {
                        int(k): int(v) for k, v in current_metrics['cluster_distribution'].items()
                    }

                # Add anomaly percentage for energy model
                if current_metrics and len(energy_labels) > 0:
                    current_metrics['anomaly_percentage'] = float(np.mean(np.array(energy_labels) == -1))

                # If no training metrics exist, use current as training baseline
                if self.validation_metrics['energy_cluster']['training'] is None:
                    self.validation_metrics['energy_cluster']['training'] = current_metrics

                self.validation_metrics['energy_cluster']['current'] = current_metrics

            return convert_numpy_types(self.validation_metrics)
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return {
                'error': str(e),
                'performance_cluster': None,
                'energy_cluster': None
            }

    def retrain_models(self):
        """Retrain models with current state data"""
        # Prepare performance data
        perf_data = []
        for machine_data in self.state.values():
            if all(k in machine_data for k in FEATURES['performance']):
                perf_data.append([machine_data[k] for k in FEATURES['performance']])

        if perf_data:
            X_perf = np.array(perf_data)
            self.performance_clf = KMeans(n_clusters=3, random_state=42)
            self.performance_clf.fit(X_perf)
            joblib.dump(self.performance_clf, PATHS['performance'])

        # Prepare energy data
        energy_data = []
        for machine_data in self.state.values():
            if all(k in machine_data for k in FEATURES['energy']):
                energy_data.append([machine_data[k] for k in FEATURES['energy']])

        if energy_data:
            X_energy = np.array(energy_data)
            X_energy_scaled = self.scaler.transform(X_energy)
            self.energy_clf = DBSCAN(eps=0.5, min_samples=5)
            self.energy_clf.fit(X_energy_scaled)
            joblib.dump(self.energy_clf, PATHS['energy'])

        print("Models retrained with latest data")

    def persist_state(self):
        """Persist current state"""
        with open(PATHS['state'], 'wb') as f:
            pickle.dump(self.state, f)
        print("Persisted state.")

    def plot_performance_clusters(self):
        """Visualize and save performance clusters with current data"""
        try:
            data = self._load_and_merge_data()
            X = data[FEATURES['performance']].values
            labels = self.performance_clf.predict(X)

            plt.figure(figsize=(12, 8))

            # 3D plot if we have 3 features
            if X.shape[1] == 3:
                ax = plt.axes(projection='3d')
                scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50)
                ax.set_xlabel(FEATURES['performance'][0])
                ax.set_ylabel(FEATURES['performance'][1])
                ax.set_zlabel(FEATURES['performance'][2])
                plt.title('Performance Clusters (3D)')
            else:
                # 2D plot if fewer features
                scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
                plt.xlabel(FEATURES['performance'][0])
                plt.ylabel(FEATURES['performance'][1])
                plt.title('Performance Clusters')

            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(VISUALIZATION_DIR, f'performance_clusters_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Performance clusters plot saved to {plot_path}")

            return plot_path

        except Exception as e:
            print(f"Error plotting performance clusters: {str(e)}")
            return None

    def plot_energy_clusters(self):
        """Visualize and save energy clusters with current data"""
        try:
            data = self._load_and_merge_data()
            X = self.scaler.transform(data[FEATURES['energy']].values)
            labels = self.energy_clf.fit_predict(X)

            plt.figure(figsize=(12, 8))

            # 3D plot if we have 3 features
            if X.shape[1] == 3:
                ax = plt.axes(projection='3d')
                scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50)
                ax.set_xlabel(FEATURES['energy'][0])
                ax.set_ylabel(FEATURES['energy'][1])
                ax.set_zlabel(FEATURES['energy'][2])
                plt.title('Energy Clusters (3D)')
            else:
                # 2D plot if fewer features
                scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
                plt.xlabel(FEATURES['energy'][0])
                plt.ylabel(FEATURES['energy'][1])
                plt.title('Energy Clusters')

            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(VISUALIZATION_DIR, f'energy_clusters_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Energy clusters plot saved to {plot_path}")

            return plot_path

        except Exception as e:
            print(f"Error plotting energy clusters: {str(e)}")
            return None

    def plot_enhanced_performance_clusters(self):
        """Improved cluster visualization for spread-out clusters"""
        try:
            data = self._load_and_merge_data()
            X = data[FEATURES['performance']].values
            labels = self.performance_clf.predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # For non-convex clusters
            spectral = SpectralClustering(n_clusters=3).fit(X)

            # For hierarchical structure
            agg = AgglomerativeClustering(n_clusters=3).fit(X)
            tsne = TSNE(n_components=2).fit_transform(X)
            plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)

            plt.figure(figsize=(18, 6))

            # Method 1: KDE Contour Plot
            plt.subplot(1, 3, 1)
            for cluster in np.unique(labels):
                if cluster == -1:
                    continue
                points = X[labels == cluster][:, :2]
                sns.kdeplot(x=points[:, 0], y=points[:, 1],
                            levels=3, thresh=0.2,
                            color=plt.cm.tab10(cluster),
                            label=f'Cluster {cluster}')
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30, alpha=0.6)
            plt.title('Density Contours of Clusters')
            plt.xlabel(FEATURES['performance'][0])
            plt.ylabel(FEATURES['performance'][1])

            # Method 2: Ellipse Fitting
            plt.subplot(1, 3, 2)
            for cluster in np.unique(labels):
                if cluster == -1:
                    continue
                points = X[labels == cluster][:, :2]
                if len(points) < 2:
                    continue

                # Calculate ellipse properties
                cov = np.cov(points.T)
                lambda_, v = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
                width, height = 2 * np.sqrt(lambda_)

                # Draw ellipse
                ell = Ellipse(xy=np.mean(points, axis=0),
                              width=width, height=height,
                              angle=angle,
                              color=plt.cm.tab10(cluster),
                              alpha=0.2)
                plt.gca().add_patch(ell)
                plt.scatter(points[:, 0], points[:, 1],
                            color=plt.cm.tab10(cluster),
                            s=30, label=f'Cluster {cluster}')
            plt.title('Ellipse-Fitted Clusters')
            plt.xlabel(FEATURES['performance'][0])

            # Method 3: Silhouette Analysis
            plt.subplot(1, 3, 3)
            if n_clusters > 1:
                silhouette_vals = silhouette_samples(X, labels)
                y_lower = 10
                for i in range(n_clusters):
                    cluster_silhouette_vals = silhouette_vals[labels == i]
                    cluster_silhouette_vals.sort()
                    y_upper = y_lower + cluster_silhouette_vals.shape[0]
                    plt.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, cluster_silhouette_vals,
                                      facecolor=plt.cm.tab10(i), alpha=0.7)
                    y_lower = y_upper + 10
                plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
                plt.title('Silhouette Analysis')
                plt.yticks([])
            else:
                plt.text(0.5, 0.5, "Need >1 cluster\nfor silhouette analysis",
                         ha='center', va='center')
                plt.axis('off')

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(VISUALIZATION_DIR, f'improved_performance_clusters_{timestamp}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()

            return plot_path

        except Exception as e:
            print(f"Error plotting clusters: {str(e)}")
            return None

    def plot_performance_clusters_enhanced(self):
        """Clean 2D performance cluster visualization with cluster stats"""
        try:
            data = self._load_and_merge_data()
            X = data[FEATURES['performance']].values
            labels = self.performance_clf.predict(X)
            centers = self.performance_clf.cluster_centers_

            # Use first two features for 2D visualization
            X_2d = X[:, :2]
            centers_2d = centers[:, :2]

            plt.figure(figsize=(16, 8))

            # Create main plot area
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot

            # Create scatter plot with cluster colors
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=80, alpha=0.7)

            # Plot cluster centers
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200,
                        linewidths=2, edgecolors='black', label='Cluster Centers')

            # Add convex hulls around clusters
            for cluster in np.unique(labels):
                if cluster != -1:  # Skip noise if using DBSCAN
                    points = X_2d[labels == cluster]
                    if len(points) >= 3:  # Need at least 3 points for convex hull
                        hull = ConvexHull(points)
                        poly = plt.Polygon(points[hull.vertices], closed=True,
                                           fill=True, alpha=0.1,
                                           color=scatter.cmap(scatter.norm(cluster)))
                        plt.gca().add_patch(poly)

            plt.xlabel(FEATURES['performance'][0], fontsize=12)
            plt.ylabel(FEATURES['performance'][1], fontsize=12)
            plt.title('Performance Clusters', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()

            # Create cluster distribution plot
            plt.subplot(1, 2, 2)  # Second plot

            # Calculate cluster counts
            unique, counts = np.unique(labels, return_counts=True)
            cluster_dist = dict(zip(unique, counts))

            # Create bar plot
            plt.bar([f"Cluster {k}" for k in cluster_dist.keys()],
                    cluster_dist.values(),
                    color=[scatter.cmap(scatter.norm(k)) for k in cluster_dist.keys()])

            # Add count labels
            for i, v in enumerate(cluster_dist.values()):
                plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)

            plt.title('Machine Distribution by Cluster', fontsize=14)
            plt.ylabel('Number of Machines', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')

            plt.tight_layout()

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(VISUALIZATION_DIR, f'performance_2d_clusters_{timestamp}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Clean performance clusters plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error plotting performance clusters: {str(e)}")
            return None

    def plot_energy_clusters_enhanced(self):
        """Clean 2D energy cluster visualization with cluster stats"""
        try:
            data = self._load_and_merge_data()
            X = self.scaler.transform(data[FEATURES['energy']].values)
            labels = self.energy_clf.fit_predict(X)

            # Use first two features for 2D visualization
            X_2d = X[:, :2]

            # For DBSCAN, calculate cluster means
            unique_labels = set(labels)
            centers = []
            for k in unique_labels:
                if k != -1:  # Skip noise
                    cluster_members = X_2d[labels == k]
                    centers.append(cluster_members.mean(axis=0))

            plt.figure(figsize=(16, 8))

            # Create main plot area
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot

            # Create scatter plot with cluster colors
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=80, alpha=0.7)

            # Plot cluster centers if they exist
            if centers:
                centers = np.array(centers)
                plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200,
                            linewidths=2, edgecolors='black', label='Cluster Centers')

            # Add convex hulls around clusters
            for cluster in unique_labels:
                if cluster != -1:  # Skip noise
                    points = X_2d[labels == cluster]
                    if len(points) >= 3:  # Need at least 3 points for convex hull
                        hull = ConvexHull(points)
                        poly = plt.Polygon(points[hull.vertices], closed=True,
                                           fill=True, alpha=0.1,
                                           color=scatter.cmap(scatter.norm(cluster)))
                        plt.gca().add_patch(poly)

            plt.xlabel(FEATURES['energy'][0], fontsize=12)
            plt.ylabel(FEATURES['energy'][1], fontsize=12)
            plt.title('Energy Clusters', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend()

            # Create cluster distribution plot
            plt.subplot(1, 2, 2)  # Second plot

            # Calculate cluster counts (including noise as -1 if present)
            unique, counts = np.unique(labels, return_counts=True)
            cluster_dist = dict(zip(unique, counts))

            # Create bar plot
            plt.bar([f"Cluster {k}" if k != -1 else "Noise" for k in cluster_dist.keys()],
                    cluster_dist.values(),
                    color=[scatter.cmap(scatter.norm(k)) if k != -1 else 'gray' for k in cluster_dist.keys()])

            # Add count labels
            for i, v in enumerate(cluster_dist.values()):
                plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)

            plt.title('Machine Distribution by Cluster', fontsize=14)
            plt.ylabel('Number of Machines', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')

            plt.tight_layout()

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(VISUALIZATION_DIR, f'energy_2d_clusters_{timestamp}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Clean energy clusters plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error plotting energy clusters: {str(e)}")
            return None

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, (np.generic, np.number)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(vars(obj))
    return obj

# Initialize model manager
model_manager = ModelManager()
# Initialize cluster models
cluster_models = ClusterModels(model_manager)

def process_mes_data(record: dict, mid: str):
    """Process manufacturing execution system data"""
    if mid not in model_manager.state:
        model_manager.state[mid] = {
            'operators': set(),
            'history': [],
            'metrics': {}
        }

        # Initialize operators set if not exists
    if 'operators' not in model_manager.state[mid]:
        model_manager.state[mid]['operators'] = set()

    if 'Units_Produced' in record and 'Defective_Units' in record:
        # Calculate metrics
        try:
            time_per_unit = record['Production_Time_min'] / record['Units_Produced']
            defect_rate = (record['Defective_Units'] / record['Units_Produced']) * 100
        except ZeroDivisionError:
            time_per_unit = 0
            defect_rate = 0

        # Update operator info
        operator_id = record['Operator_ID']
        model_manager.state[mid]['operators'].add(operator_id)

        # Update affinity scores
        key = (operator_id, mid)
        current_rate = model_manager.affinity_scores.get(key, 0.5)
        new_rate = 1 - (record['Defective_Units'] / record['Units_Produced'])
        model_manager.affinity_scores[key] = (current_rate + new_rate) / 2

        # Store metrics
        model_manager.state[mid].update({
            'Time_per_Unit': time_per_unit,
            'Defect_Rate': defect_rate,
            'Affinity_Score': model_manager.affinity_scores[key],
            'last_operator': operator_id
        })


def process_energy_data(record: dict, mid: str):
    """Process energy-related data"""
    if 'Power_Consumption_kW' in record and 'Units_Produced' in model_manager.state[mid]:
        energy_per_unit = record['Power_Consumption_kW'] / model_manager.state[mid]['Units_Produced']
        model_manager.state[mid]['Energy_per_Unit'] = energy_per_unit


def kafka_consumer_loop():
    consumer = KafkaConsumer(
        'mes-stream', 'scada-stream', 'iot-stream',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        group_id='online-performance-energy',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    print("Kafka consumer started...")

    message_count = 0
    for message in consumer:
        record = message.value
        mid = record.get('Machine_ID')
        if not mid:
            continue

        if mid not in model_manager.state:
            model_manager.state[mid] = {'operators': set(), 'history': [], 'metrics': {}}

        if mid not in cluster_models.state:
            cluster_models.state[mid] = {'history': []}

        # Update state
        model_manager.state[mid].update(record)

        if "Alarm_Code" in record and record['Alarm_Code'] == "":
            record['Alarm_Code'] = "None"

        # Process MES data
        if message.topic == 'mes-stream':
            process_mes_data(record, mid)

        # Process energy data
        elif message.topic in ['scada-stream', 'iot-stream']:
            process_energy_data(record, mid)

        # Store message
        model_manager.state[mid]['history'].append({
            'timestamp': datetime.now(),
            'topic': message.topic,
            'data': record
        })

        # Update state
        cluster_models.state[mid].update(record)
        cluster_models.state[mid]['history'].append({
            'timestamp': datetime.now(),
            'topic': message.topic,
            'data': record
        })

        # Update affinity scores if MES data
        if message.topic == 'mes-stream' and 'Operator_ID' in record:
            operator_id = record['Operator_ID']
            key = (operator_id, mid)
            current = cluster_models.state.get('affinity_scores', {}).get(key, 0.5)
            new_rate = 1 - (record.get('Defective_Units', 0) / record.get('Units_Produced', 1))
            cluster_models.state.setdefault('affinity_scores', {})[key] = (current + new_rate) / 2

            # Periodically retrain
        message_count += 1
        if message_count % 50 == 0:
            model_manager.retrain_models()
            cluster_models.update_cluster_models()
            cluster_models.persist_models()
            model_manager.validate_models()
            # Optionally log or store these metrics
            print(f"Model validation metrics: {model_manager.validation_metrics}")

        # model_manager.persist_state()

# Start Kafka consumer thread
threading.Thread(target=kafka_consumer_loop, daemon=True).start()

# FastAPI App
app = FastAPI()


class MachineRequest(BaseModel):
    machine_id: str

class PerformanceAnalysisResponse(BaseModel):
    machine_id: str
    current_tier: str
    metrics: dict
    cluster_center: List[float]
    top_operators: List[dict]
    improvement_targets: dict

class ForecastRequest(BaseModel):
    machine_id: str
    hours_ahead: int = 1
    operator_id: Optional[int] = None

@app.get("/")
def health_check():
    return {"status": "running", "version": "1.0"}

@app.post('/performance-cluster')
def get_performance_cluster(req: MachineRequest):
    mid = req.machine_id
    if mid not in model_manager.state or not all(k in model_manager.state[mid] for k in FEATURES['performance']):
        raise HTTPException(404, detail="Machine data not available")

    features = np.array([[model_manager.state[mid][k] for k in FEATURES['performance']]])
    cluster = model_manager.performance_clf.predict(features)[0]

    return {
        'machine_id': mid,
        'performance_cluster': int(cluster),
        'cluster_center': model_manager.performance_clf.cluster_centers_[cluster].tolist()
    }

@app.post('/energy-cluster')
async def get_energy_cluster(req: MachineRequest):
    """Get energy cluster with proper type conversion"""
    try:
        mid = req.machine_id
        if mid not in model_manager.state or not all(k in model_manager.state[mid] for k in FEATURES['energy']):
            raise HTTPException(404, detail="Machine data not available")

        # Prepare features and convert to native Python types
        features = np.array([
            [
                float(model_manager.state[mid]['Energy_per_Unit']),
                float(model_manager.state[mid]['Power_Consumption_kW']),
                float(model_manager.state[mid]['Temperature_C'])
            ]
        ])

        # Scale features
        features_scaled = model_manager.scaler.transform(features)

        # Predict cluster
        cluster = model_manager.energy_clf.fit_predict(features_scaled)[0]

        # Convert all NumPy types to native Python types
        response = {
            "machine_id": mid,
            "energy_cluster": int(cluster),
            "is_anomaly": bool(cluster == -1),  # Explicit bool conversion
            "features": {
                "energy_per_unit": float(model_manager.state[mid]['Energy_per_Unit']),
                "power_consumption_kw": float(model_manager.state[mid]['Power_Consumption_kW']),
                "temperature_c": float(model_manager.state[mid]['Temperature_C'])
            }
        }

        return response

    except Exception as e:
        raise HTTPException(500, detail=f"Energy cluster analysis failed: {str(e)}")

@app.post('/optimal-pairings')
def get_optimal_pairings(req: MachineRequest):
    mid = req.machine_id
    if mid not in model_manager.state:
        raise HTTPException(status_code=404, detail="Machine not found")

    # Get all operators who have worked with this machine
    operators = list(model_manager.state[mid]['operators'])
    if not operators:
        return {'machine_id': mid, 'optimal_pairings': []}

    # Calculate scores for each operator
    pairings = []
    for op in operators:
        key = (op, mid)
        score = model_manager.affinity_scores.get(key, 0.5)
        pairings.append({
            'operator_id': op,
            'affinity_score': score,
            'recommended': score >= 0.8  # Threshold for recommendation
        })

    # Sort by highest affinity
    pairings.sort(key=lambda x: x['affinity_score'], reverse=True)

    print(f"Optimal pairings calculated for {mid}: {len(pairings)} operators")
    return {'machine_id': mid, 'optimal_pairings': pairings}


@app.post('/energy-efficiency')
def get_energy_efficiency(req: MachineRequest):
    mid = req.machine_id
    if mid not in model_manager.state or 'Energy_per_Unit' not in model_manager.state[mid]:
        raise HTTPException(status_code=404, detail="Energy data not available")

    # Calculate peak demand ratio (simplified)
    history = model_manager.state[mid].get('history', [])
    if len(history) > 1:
        power_values = [h['data'].get('Power_Consumption_kW', 0) for h in history]
        peak_ratio = max(power_values) / (sum(power_values) / len(power_values))
    else:
        peak_ratio = 1.0

    efficiency = 1 / model_manager.state[mid]['Energy_per_Unit']

    print(f"Energy efficiency for {mid}: "
          f"{model_manager.state[mid]['Energy_per_Unit']:.2f} kW/unit, "
          f"peak ratio: {peak_ratio:.2f}, "
          f"efficiency: {efficiency:.2f}")

    return {
        'machine_id': mid,
        'energy_per_unit': model_manager.state[mid]['Energy_per_Unit'],
        'peak_demand_ratio': peak_ratio,
        'efficiency_score': efficiency  # Higher is better
    }

# Additional endpoints would follow the same pattern...
@app.post("/performance-analysis", response_model=PerformanceAnalysisResponse)
def performance_analysis(request: MachineRequest):
    """
    Get comprehensive performance analysis including:
    - Current performance tier
    - Recommended operators
    - Improvement targets
    """
    try:
        return model_manager.analyze_performance(request.machine_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/forecast-energy-consumption')
async def forecast_energy_consumption(req: ForecastRequest):
    """Energy consumption forecast using combined SCADA and IoT data"""
    try:
        # Validate machine exists
        if req.machine_id not in cluster_models.state:
            raise HTTPException(404, detail={
                "error": "machine_not_found",
                "message": f"Machine {req.machine_id} not monitored",
                "solution": "Check machine ID or ensure data is being collected"
            })

        machine_data = cluster_models.state[req.machine_id]
        current_time = datetime.now()

        # 1. LOAD AND MERGE HISTORICAL DATA -----------------------
        try:
            # Load both SCADA and IoT data
            scada = pd.read_csv('data/generated_data/historical-scada.csv', parse_dates=['Timestamp'])
            iot = pd.read_csv('data/generated_data/historical-iot.csv', parse_dates=['Timestamp'])

            # Merge datasets on Timestamp and Machine_ID
            historical = pd.merge(
                scada,
                iot,
                on=['Timestamp', 'Machine_ID'],
                how='inner'
            )

            # Filter for this machine
            historical = historical[historical['Machine_ID'] == req.machine_id]

            if historical.empty:
                raise HTTPException(400, detail={
                    "error": "no_historical_data",
                    "message": f"No combined historical data for {req.machine_id}",
                    "required_sources": ["historical-scada.csv", "historical-iot.csv"],
                    "solution": "Verify both data files contain records for this machine"
                })

        except Exception as e:
            raise HTTPException(400, detail={
                "error": "data_load_failed",
                "message": f"Could not load historical data: {str(e)}",
                "solution": "Verify SCADA and IoT data files exist and are properly formatted"
            })

        # 2. FEATURE EXTRACTION -----------------------------------
        features = []
        power_values = []
        valid_records = 0

        for _, row in historical.iterrows():
            try:
                features.append([
                    row['Timestamp'].hour,  # Hour of day
                    row['Temperature_C'],
                    row['Vibration_mm_s']
                ])
                power_values.append(row['Power_Consumption_kW'])
                valid_records += 1
            except KeyError as e:
                print(f"Missing expected column in historical data: {str(e)}")
                continue
            except Exception as e:
                print(f"Skipping corrupt record: {str(e)}")
                continue

        if valid_records < 10:
            raise HTTPException(400, detail={
                "error": "insufficient_samples",
                "message": f"Only found {valid_records} valid historical records",
                "minimum_required": 10,
                "solution": "Ensure machine has sufficient operational history in both SCADA and IoT data"
            })

        # 3. MODEL TRAINING & FORECASTING -------------------------
        try:
            X = np.array(features)
            X_scaled = cluster_models.energy_scaler.fit_transform(X)
            clusters = cluster_models.energy_cluster.fit_predict(X_scaled)

            # Create cluster statistics using percentiles
            cluster_stats = {
                cluster: {
                    'median': np.median([power_values[i] for i, c in enumerate(clusters) if c == cluster]),
                    'q10': np.percentile([power_values[i] for i, c in enumerate(clusters) if c == cluster], 10),
                    'q90': np.percentile([power_values[i] for i, c in enumerate(clusters) if c == cluster], 90)
                }
                for cluster in set(clusters)
            }

        except Exception as e:
            raise HTTPException(500, detail={
                "error": "model_training_failed",
                "message": f"Could not train forecasting model: {str(e)}",
                "solution": "Check model configuration and data quality"
            })

        # 4. GENERATE FORECASTS -----------------------------------
        response = {
            "machine_id": req.machine_id,
            "current_metrics": {
                "power_kW": machine_data.get('Power_Consumption_kW'),
                "temperature_C": machine_data.get('Temperature_C'),
                "vibration_mm_s": machine_data.get('Vibration_mm_s'),
                "timestamp": current_time.isoformat()
            },
            "forecasts": [],
            "data_quality": {
                "historical_samples": valid_records,
                "clusters_available": len(cluster_stats),
                "data_sources": ["SCADA", "IoT"]
            }
        }

        future_times = [current_time + timedelta(hours=h) for h in range(1, req.hours_ahead + 1)]

        for hours_ahead, ts in enumerate(future_times, 1):
            try:
                # Use current sensor readings as proxies for future values
                features = np.array([[
                    ts.hour,
                    machine_data.get('Temperature_C', 25),
                    machine_data.get('Vibration_mm_s', 0)
                ]])
                features_scaled = cluster_models.energy_scaler.transform(features)
                cluster = int(cluster_models.energy_cluster.predict(features_scaled)[0])

                stats = cluster_stats.get(cluster, {})
                response["forecasts"].append({
                    "hours_ahead": hours_ahead,
                    "timestamp": ts.isoformat(),
                    "predicted_cluster": cluster,
                    "predicted_power_kW": float(stats.get('median', 0)),
                    "confidence_band": {
                        "lower": float(stats.get('q10', 0)),
                        "upper": float(stats.get('q90', 0))
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to generate forecast for hour {hours_ahead}: {str(e)}")
                continue

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail={
            "error": "unexpected_error",
            "message": str(e),
            "solution": "Check server logs and data files"
        })


@app.post('/forecast-performance-trend')
async def forecast_performance_trend(req: ForecastRequest):
    """Performance trend forecast using ModelManager's models"""
    try:
        # Input validation
        if not req.operator_id:
            raise HTTPException(400, detail="Operator ID required")
        if req.machine_id not in model_manager.state:
            raise HTTPException(404, detail="Machine not found")

        # Get machine data
        machine_data = model_manager.state[req.machine_id]

        # Calculate required metrics if not present
        if 'Time_per_Unit' not in machine_data or 'Defect_Rate' not in machine_data:
            # Try to calculate from history
            if 'history' in machine_data:
                last_mes_record = None
                for record in reversed(machine_data['history']):
                    if record.get('topic') == 'mes-stream' and 'data' in record:
                        last_mes_record = record['data']
                        break

                if last_mes_record:
                    try:
                        units = last_mes_record.get('Units_Produced', 1)
                        time_per_unit = last_mes_record.get('Production_Time_min', 0) / units
                        defect_rate = last_mes_record.get('Defective_Units', 0) / units

                        machine_data['Time_per_Unit'] = time_per_unit
                        machine_data['Defect_Rate'] = defect_rate
                    except ZeroDivisionError:
                        raise HTTPException(400, detail="Cannot calculate metrics - production data invalid")

        # Verify required metrics exist
        required_metrics = ['Time_per_Unit', 'Defect_Rate']
        if not all(m in machine_data for m in required_metrics):
            missing = [m for m in required_metrics if m not in machine_data]
            raise HTTPException(400, detail=f"Missing performance metrics: {missing}")

        # Verify operator exists
        operator_key = (int(req.operator_id), req.machine_id)
        if operator_key not in model_manager.affinity_scores:
            raise HTTPException(400, detail="No history for this operator-machine pair")

        # Prepare features
        future_times = [datetime.now() + timedelta(hours=h) for h in range(1, req.hours_ahead + 1)]
        features = []

        for ts in future_times:
            hour = ts.hour
            time_per_unit = machine_data['Time_per_Unit']
            defect_rate = machine_data['Defect_Rate']
            affinity = model_manager.affinity_scores.get(operator_key, 0.5)

            features.append([hour, defect_rate, affinity])

        if not features:
            raise HTTPException(400, detail="Could not prepare features")

        # Scale features and predict
        try:
            features_scaled = model_manager.scaler.transform(features)
            clusters = model_manager.performance_clf.predict(features_scaled)
            cluster_centers = model_manager.performance_clf.cluster_centers_
        except Exception as e:
            raise HTTPException(500, detail=f"Model prediction failed: {str(e)}")

        return {
            "machine_id": req.machine_id,
            "operator_id": req.operator_id,
            "current_performance": {
                "time_per_unit": float(machine_data['Time_per_Unit']),
                "defect_rate": float(machine_data['Defect_Rate']),
                "affinity_score": float(model_manager.affinity_scores[operator_key])
            },
            "forecasts": [
                {
                    "hours_ahead": h + 1,
                    "timestamp": future_times[h].isoformat(),
                    "predicted_time_per_unit": float(cluster_centers[clusters[h]][0]),
                    "predicted_defect_rate": float(cluster_centers[clusters[h]][1]),
                    "hour_of_day": future_times[h].hour,
                    "cluster_id": int(clusters[h])
                }
                for h in range(req.hours_ahead)
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post('/detect-energy-anomalies')
def detect_energy_anomalies(req: MachineRequest):
    """
    Enhanced anomaly detection with:
    - Better data validation
    - More detailed diagnostics
    - Alternative calculation methods
    - Practical workarounds
    """
    try:
        # 1. Machine Validation -----------------------------------
        if req.machine_id not in model_manager.state:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "machine_not_found",
                    "message": f"Machine {req.machine_id} not monitored",
                    "solution": "Check machine ID or verify data collection is enabled"
                }
            )

        machine_data = model_manager.state[req.machine_id]

        # 2. Data Availability Check ------------------------------
        if 'history' not in machine_data or not machine_data['history']:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "no_history",
                    "message": "No historical records found",
                    "diagnostics": {
                        "total_records": 0,
                        "record_types": [],
                        "last_update": None
                    },
                    "solution": "Ensure machine is sending data to all streams"
                }
            )

        # 3. Data Extraction with Enhanced Logging ----------------
        energy_readings = []
        record_sources = set()
        last_timestamps = []

        for record in machine_data['history'][-48:]:  # Check last 48 records (24 hours at 30-min intervals)
            try:
                if ('data' in record and
                        isinstance(record['data'], dict) and
                        'Power_Consumption_kW' in record['data']):
                    energy_readings.append(float(record['data']['Power_Consumption_kW']))
                    record_sources.add(record.get('topic', 'unknown'))
                    last_timestamps.append(record.get('timestamp'))
            except (KeyError, TypeError, ValueError) as e:
                continue

        # 4. Flexible Analysis Based on Available Data ------------
        if len(energy_readings) < 5:  # Absolute minimum for any analysis
            # Try to get default values from machine specs if available
            default_power = machine_data.get('specs', {}).get('rated_power_kW')

            return {
                "machine_id": req.machine_id,
                "status": "insufficient_data",
                "available_readings": len(energy_readings),
                "required_readings": 10,
                "record_sources": list(record_sources),
                "last_data_points": sorted(last_timestamps)[-3:] if last_timestamps else None,
                "default_power_rating": default_power,
                "suggestion": "Check SCADA data pipeline configuration",
                "temporary_workaround": {
                    "using_default_values": default_power is not None,
                    "estimated_normal_range": [
                        default_power * 0.9 if default_power else None,
                        default_power * 1.1 if default_power else None
                    ] if default_power else None
                }
            }

        # 5. Robust Statistical Analysis -------------------------
        current_value = float(energy_readings[-1])
        historical_window = energy_readings[-24:] if len(energy_readings) >= 24 else energy_readings

        # Handle small sample sizes differently
        if len(historical_window) < 10:
            # Use median-based detection for small samples
            median_val = float(np.median(historical_window))
            mad = 1.4826 * float(
                np.median(np.abs(np.array(historical_window) - median_val)))  # Median Absolute Deviation
            modified_z = 0.6745 * (current_value - median_val) / mad if mad > 0 else 0
            anomaly_score = min(1.0, max(0.0, 0.5 + (abs(modified_z) / 3)))
        else:
            # Standard deviation-based for larger samples
            mean_val = float(np.mean(historical_window))
            std_val = float(np.std(historical_window))
            z_score = (current_value - mean_val) / std_val if std_val > 0 else 0
            anomaly_score = min(1.0, max(0.0, 0.5 + (abs(z_score) / 3)))

            is_anomaly = anomaly_score > 0.8

            # 6. Enhanced Response ------------------------------------
        return {
            "machine_id": req.machine_id,
            "current_consumption_kw": current_value,
            "analysis_window": {
                "start": last_timestamps[0] if last_timestamps else None,
                "end": last_timestamps[-1] if last_timestamps else None,
                "hours_covered": (last_timestamps[-1] - last_timestamps[0]).total_seconds() / 3600
                if len(last_timestamps) > 1 else 0
            },
            "statistical_metrics": {
                "mean_kw": float(np.mean(historical_window)) if len(historical_window) >= 5 else None,
                "median_kw": float(np.median(historical_window)),
                "std_dev_kw": float(np.std(historical_window)) if len(historical_window) >= 5 else None,
                "min_kw": float(min(historical_window)),
                "max_kw": float(max(historical_window)),
                "data_points": len(historical_window)
            },
            "anomaly_detection": {
                "score": anomaly_score,
                "threshold": 0.8,
                "is_anomaly": is_anomaly,
                "method": "MAD" if len(historical_window) < 10 else "Z-score",
                "confidence": min(0.99, max(0.5, anomaly_score * 1.2))  # Adjusted confidence
            },
            "recommendations": [
                {
                    "priority": "high" if is_anomaly else "low",
                    "action": "Immediate inspection recommended" if is_anomaly
                    else "Routine monitoring sufficient",
                    "triggers": [
                        f"Power consumption {current_value:.1f}kW vs expected range "
                        f"[{mean_val - 2 * std_val:.1f}-{mean_val + 2 * std_val:.1f}]kW"
                    ] if len(historical_window) >= 10 else ["Limited historical data"]
                }
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "context": {
                    "machine_id": req.machine_id,
                    "processing_stage": "anomaly_detection"
                }
            }
        )


@app.post('/model-metrics')
def get_model_metrics():
    """Get comprehensive model performance metrics with insights"""
    try:
        metrics = model_manager.validate_models()

        if 'error' in metrics:
            raise HTTPException(500, detail=metrics['error'])

        # Generate performance insights
        performance_insights = generate_performance_insights(
            metrics.get('performance_cluster', {})
        )

        # Generate energy insights
        energy_insights = generate_energy_insights(
            metrics.get('energy_cluster', {})
        )

        # Prepare response and convert numpy types
        response = {
            "performance_model": {
                "metrics": metrics.get('performance_cluster', {}),
                "insights": performance_insights,
                "health_status": assess_cluster_health(
                    metrics.get('performance_cluster', {}).get('current', {})
                )
            },
            "energy_model": {
                "metrics": metrics.get('energy_cluster', {}),
                "insights": energy_insights,
                "health_status": assess_cluster_health(
                    metrics.get('energy_cluster', {}).get('current', {}),
                    is_energy_model=True
                )
            },
            "timestamp": datetime.now().isoformat(),
            "model_versions": {
                "performance": "1.0",
                "energy": "1.0"
            }
        }

        # Convert all numpy types in the response
        return convert_numpy_types(response)

    except Exception as e:
        raise HTTPException(500, detail=f"Failed to validate models: {str(e)}")


# Helper functions for generating insights
def generate_performance_insights(metrics):
    """Generate human-readable insights for performance model"""
    current = metrics.get('current', {})
    training = metrics.get('training', {})

    insights = []

    if current.get('silhouette') is not None:
        sil_diff = (current['silhouette'] - training['silhouette']) if training else 0
        insights.append({
            "metric": "Cluster Separation",
            "value": current['silhouette'],
            "interpretation": interpret_silhouette(current['silhouette']),
            "trend": "improving" if sil_diff > 0 else "declining" if sil_diff < 0 else "stable",
            "change": f"{abs(sil_diff):.2f}" if training else "N/A"
        })

    if current.get('davies_bouldin') is not None:
        db_diff = (current['davies_bouldin'] - training['davies_bouldin']) if training else 0
        insights.append({
            "metric": "Cluster Compactness",
            "value": current['davies_bouldin'],
            "interpretation": interpret_davies_bouldin(current['davies_bouldin']),
            "trend": "improving" if db_diff < 0 else "declining" if db_diff > 0 else "stable",
            "change": f"{abs(db_diff):.2f}" if training else "N/A"
        })

    if 'cluster_distribution' in metrics:
        insights.append({
            "metric": "Cluster Distribution",
            "value": metrics['cluster_distribution'],
            "interpretation": "Distribution of data across clusters"
        })

    return insights


def generate_energy_insights(metrics):
    """Generate human-readable insights for energy model"""
    current = metrics.get('current', {})
    training = metrics.get('training', {})

    insights = []

    if current.get('silhouette') is not None:
        sil_diff = (current['silhouette'] - training['silhouette']) if training else 0
        insights.append({
            "metric": "Cluster Separation",
            "value": current['silhouette'],
            "interpretation": interpret_silhouette(current['silhouette']),
            "trend": "improving" if sil_diff > 0 else "declining" if sil_diff < 0 else "stable",
            "change": f"{abs(sil_diff):.2f}" if training else "N/A"
        })

    if current.get('anomaly_percentage') is not None:
        anom_diff = (current['anomaly_percentage'] - training['anomaly_percentage']) if training else 0
        insights.append({
            "metric": "Anomaly Rate",
            "value": f"{current['anomaly_percentage']:.1%}",
            "interpretation": interpret_anomaly_rate(current['anomaly_percentage']),
            "trend": "increasing" if anom_diff > 0 else "decreasing" if anom_diff < 0 else "stable",
            "change": f"{abs(anom_diff):.1%}" if training else "N/A"
        })

    if 'cluster_distribution' in metrics:
        insights.append({
            "metric": "Cluster Distribution",
            "value": metrics['cluster_distribution'],
            "interpretation": "Distribution of data across clusters including anomalies"
        })

    return insights


# Interpretation helpers
def interpret_silhouette(score):
    if score > 0.7:
        return "Strong cluster structure"
    elif score > 0.5:
        return "Reasonable cluster structure"
    elif score > 0.25:
        return "Weak cluster structure"
    else:
        return "No substantial cluster structure"


def interpret_davies_bouldin(score):
    if score < 0.5:
        return "Excellent cluster separation"
    elif score < 1.0:
        return "Good cluster separation"
    elif score < 1.5:
        return "Moderate cluster separation"
    else:
        return "Poor cluster separation"


def interpret_anomaly_rate(rate):
    if rate < 0.01:
        return "Very low anomaly rate"
    elif rate < 0.05:
        return "Normal operating range"
    elif rate < 0.1:
        return "Elevated anomaly rate - investigate"
    else:
        return "High anomaly rate - immediate attention needed"


def assess_cluster_health(metrics, is_energy_model=False):
    """Assess overall model health"""
    if not metrics:
        return "UNKNOWN"

    health_score = 0
    weights = {
        'silhouette': 0.4,
        'davies_bouldin': 0.4,
        'anomaly_percentage': 0.2 if is_energy_model else 0
    }

    # Calculate weighted health score
    if 'silhouette' in metrics:
        health_score += weights['silhouette'] * min(1, max(0, metrics['silhouette'] / 0.7))

    if 'davies_bouldin' in metrics:
        health_score += weights['davies_bouldin'] * min(1, max(0, (2 - metrics['davies_bouldin']) / 2))

    if is_energy_model and 'anomaly_percentage' in metrics:
        health_score += weights['anomaly_percentage'] * min(1, max(0, 1 - (metrics['anomaly_percentage'] / 0.1)))

    # Determine health status
    if health_score > 0.8:
        return "HEALTHY"
    elif health_score > 0.6:
        return "STABLE"
    elif health_score > 0.4:
        return "DEGRADED"
    else:
        return "CRITICAL"

@app.post('/retrain-models')
def retrain_models():
    """Retrain models and return validation metrics"""
    try:
        model_manager._train_performance_model()
        model_manager._train_energy_model()
        metrics = model_manager.validate_models()

        return {
            "status": "retraining_complete",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Retraining failed: {str(e)}")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)