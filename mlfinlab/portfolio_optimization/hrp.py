import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.covariance import OAS


class HierarchicalRiskParity:
    '''
    This module implements the HRP algorithm mentioned in the following paper:
    López de Prado, Marcos, Building Diversified Portfolios that Outperform Out-of-Sample (May 23, 2016).
    Journal of Portfolio Management, 2016;
    The code is reproduced with modification from his book: Advances in Financial Machine Learning
    '''

    def __init__(self):
        return

    @staticmethod
    def _tree_clustering(self, correlation, method='single'):
        '''
        Perform the traditional heirarchical tree clustering

        :param correlation: (np.array) correlation matrix of the assets
        :param method: (str) the type of clustering to be done
        :return: distance matrix and clusters
        '''

        distances = np.sqrt((1 - correlation).round(5) / 2)
        clusters = linkage(squareform(distances.values), method=method)
        return distances, clusters

    @staticmethod
    def _quasi_diagnalization(self, N, curr_index):
        '''
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param N: (int) index of element in the cluster list
        :param curr_index: (int) current index
        :return: (list) the assets rearranged according to hierarchical clustering
        '''

        if curr_index < N:
            return [curr_index]

        left = int(self.clusters[curr_index - N, 0])
        right = int(self.clusters[curr_index - N, 1])

        return (self._quasi_diagnalization(N, left) + self._quasi_diagnalization(N, right))

    @staticmethod
    def _get_seriated_matrix(self, assets, distances, correlations):
        '''
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param assets:
        :param distances:
        :param correlations:
        :return: (np.array) re-arranged distance matrix based on tree clusters
        '''

        ordering = assets[self.ordered_indices]
        seriated_distances = distances.loc[ordering, ordering]
        seriated_correlations = correlations.loc[ordering, ordering]
        return seriated_distances, seriated_correlations

    @staticmethod
    def _recursive_bisection(self, covariances, assets):
        '''
        Recursively assign weights to the clusters - ultimately assigning weights to the inidividual assets

        :param covariances: (np.array) the covariance matrix
        :param assets:
        '''

        self.weights = pd.Series(1, index=self.ordered_indices)
        clustered_alphas = [self.ordered_indices]

        while len(clustered_alphas) > 0:
            clustered_alphas = [cluster[start:end]
                                for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]

            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                # Get left cluster variance
                left_subcovar = covariances.iloc[left_cluster, left_cluster]
                inv_diag = 1 / np.diag(left_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                # Get right cluster variance
                right_subcovar = covariances.iloc[right_cluster, right_cluster]
                inv_diag = 1 / np.diag(right_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                # Calculate allocation factor and weights
                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)
                self.weights[left_cluster] *= alloc_factor
                self.weights[right_cluster] *= 1 - alloc_factor

        # Assign actual asset values to weight index
        self.weights.index = assets[self.ordered_indices]
        self.weights = pd.DataFrame(self.weights)
        self.weights = self.weights.T

    def plot_clusters(self, height=10, width=10):
        '''
        Plot a dendrogram of the hierarchical clusters

        :param height: (int) height of the plot
        :param width: (int) width of the plot
        '''
        plt.figure(figsize=(width, height))
        dendrogram(self.clusters)
        plt.show()

    @staticmethod
    def _calculate_returns(self, asset_prices, resample_returns_by):
        '''Calculate the annualised mean historical returns from asset price data


        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param resample_returns_by: (str) specifies how to resample the returns - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :return: (pd.Dataframe) stock returns
        '''

        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        asset_returns = asset_returns.resample(resample_returns_by).mean()
        return asset_returns

    @staticmethod
    def _shrink_covariance(self, covariance):
        '''
        Regularise/Shrink the asset covariances

        :param covariance: (pd.Dataframe) asset returns covariances
        :return: (pd.Dataframe) shrinked asset returns covariances
        '''

        oas = OAS()
        oas.fit(covariance)
        shrinked_covariance = oas.covariance_
        return pd.DataFrame(shrinked_covariance, index=covariance.columns, columns=covariance.columns)

    @staticmethod
    def _cov2corr(self, covariance):
        '''
        Calculate the correlations from asset returns covariance matrix

        :param covariance: (pd.Dataframe) asset returns covariances
        :return: (pd.Dataframe) correlations between asset returns
        '''

        D = np.zeros_like(covariance)
        d = np.sqrt(np.diag(covariance))
        np.fill_diagonal(D, d)
        DInv = np.linalg.inv(D)
        corr = np.dot(np.dot(DInv, covariance), DInv)
        corr = pd.DataFrame(corr, index=covariance.columns, columns=covariance.columns)
        return corr

    @staticmethod
    def allocate(self, asset_prices, resample_returns_by='B', use_shrinkage=False):
        '''
        Calculate asset allocations using HRP algorithm

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
                                            indexed by date
        :param resample_returns_by: (str) specifies how to resample the returns - weekly, daily, monthly etc.. Defaults to
                                          'B' meaning daily business days which is equivalent to no resampling
        :param use_shrinkage: (Boolean) specifies whether to shrink the covariances
        '''

        if type(asset_prices) != pd.DataFrame:
            raise ValueError("Asset prices matrix must be a dataframe")
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

        # Calculate the returns
        asset_returns = self._calculate_returns(asset_prices, resample_returns_by=resample_returns_by)

        N = asset_returns.shape[1]
        assets = asset_returns.columns

        # Covariance and correlation
        cov = asset_returns.cov()
        if use_shrinkage:
            cov = self._shrink_covariance(covariance=cov)
        corr = self._cov2corr(covariance=cov)

        # Step-1: Tree Clustering
        distances, self.clusters = self._tree_clustering(correlation=corr)

        # Step-2: Quasi Diagnalization
        self.ordered_indices = self._quasi_diagnalization(N, 2 * N - 2)
        self.seriated_distances, self.seriated_correlations = self._get_seriated_matrix(assets=assets,
                                                                                        distances=distances,
                                                                                        correlations=corr)

        # Step-3: Recursive Bisection
        self._recursive_bisection(covariances=cov, assets=assets)