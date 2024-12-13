import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

class SyntheticRealisticDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

class KFoldDataModule(pl.LightningDataModule):
    def __init__(self, num_samples, num_features, test_size, k_folds, current_fold, seed=42, batch_size=1024, num_workers=8, persistent_workers=True):
        super().__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.test_size = test_size
        self.k_folds = k_folds
        self.current_fold = current_fold
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        np.random.seed(self.seed)

        n = self.num_samples
        d = self.num_features

        # Distributions and complexity
        n_normal = 100
        n_uniform = 50
        n_exp = 30
        n_categ = 5  # 5 categorical features => 20 one-hot after encoding

        X_normal = np.random.randn(n, n_normal)
        X_uniform = np.random.rand(n, n_uniform)*10-5
        X_exp = np.random.exponential(scale=1.0, size=(n,n_exp))

        cat_data = np.random.randint(0,4,size=(n,n_categ))
        cat_onehot = []
        for i in range(n_categ):
            oh = np.zeros((n,4))
            oh[np.arange(n), cat_data[:,i]] = 1
            cat_onehot.append(oh)
        cat_onehot = np.concatenate(cat_onehot, axis=1) # n x 20

        X = np.concatenate([X_normal, X_uniform, X_exp, cat_onehot], axis=1)

        # Missing values in first 50 numeric features
        missing_mask = (np.random.rand(n,50)<0.05)
        X[:,:50][missing_mask]=np.nan
        for col in range(50):
            col_data = X[:,col]
            mean_val = np.nanmean(col_data)
            col_data[np.isnan(col_data)] = mean_val
            X[:,col] = col_data

        # Outliers in features [50:60]
        outlier_mask = (np.random.rand(n,10)<0.005)
        X[:,50:60][outlier_mask]*=50

        # Complex score
        score = np.sum(X[:,:20],axis=1)
        score += np.sum(np.sin(X[:,20:30]),axis=1)
        score += np.sum(np.cos(X[:,30:40]),axis=1)
        score += np.sum(X[:,40:50]**2,axis=1)
        score += np.sum(X[:,180:190]*X[:,50:60],axis=1)
        score += np.random.normal(0,3.0,n)

        # Label
        y = (score>0).astype('float32')

        # Reduce label noise to 0.5%
        flip_mask = (np.random.rand(n)<0.005)
        y[flip_mask]=1-y[flip_mask]

        X_trainval,X_test,y_trainval,y_test = train_test_split(
            X,y,test_size=self.test_size,random_state=self.seed,stratify=y
        )

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        folds = list(skf.split(np.arange(len(X_trainval)), y_trainval))
        train_indices,val_indices=folds[self.current_fold]

        self.train_dataset = SyntheticRealisticDataset(X_trainval[train_indices], y_trainval[train_indices])
        self.val_dataset = SyntheticRealisticDataset(X_trainval[val_indices], y_trainval[val_indices])
        self.test_dataset = SyntheticRealisticDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
