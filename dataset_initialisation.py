import numpy as np
from sklearn.mixture import GaussianMixture

class GMM_Init():

    def __init__(self, dataset, n_components=2):
        self.dataset = dataset

        # gmm = GaussianMixture(n_components=n_components, reg_covar = 0.085, covariance_type='full')
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.fitted_gmm = gmm.fit(dataset)

        self.cluster_centres = self.fitted_gmm.means_
        self.cluster_covs = self.fitted_gmm.covariances_

        self.gamma_estimates = np.array([self.gamma_scaled(cov_mat) for cov_mat in self.cluster_covs])

        self.labels = gmm.predict(dataset)
    

    def mu_prior_cov_estimate(self):
        cluster_centres = self.fitted_gmm.means_

        # print(cluster_centres, "cluster centres")

        mu_cov = np.cov(cluster_centres.T)

        return np.diag(np.diag(mu_cov))
    
    def gamma_scaled(self, cov_mat):
        ν = cov_mat[-1,-1]
        γ = cov_mat[-1,:-1] / np.sqrt(ν)
        return γ

    def gamma_prior_cov_estimate(self):
        gamma_cov_estimate = np.cov(self.gamma_estimates.T)
        
        # Check if gamma_cov_estimate is a scalar or a 0-dimensional array
        if np.ndim(gamma_cov_estimate) == 0 or gamma_cov_estimate.shape == ():
            gamma_cov_estimate = np.array([[gamma_cov_estimate]])
        else:
            gamma_cov_estimate = np.diag(np.diag(gamma_cov_estimate))

        return gamma_cov_estimate


    def sigma_prior_params_estimate(self):
        d = len(self.cluster_covs[0])

        sigma_star_estimates = []

        for i, cov_mat in enumerate(self.cluster_covs):
            sigma_star_estimates.append(cov_mat[:-1,:-1] - np.outer(self.gamma_estimates[i], self.gamma_estimates[i]))
        
        sigma_star_estimates = np.array(sigma_star_estimates)

        self.sigma_star_estimates = sigma_star_estimates

        dim = len(sigma_star_estimates[0])
        dof = d+3 # Set it to be d+3 for now
        sigma_star_mean = np.mean(sigma_star_estimates, axis = 0)
        scale = sigma_star_mean * (dof - (dim + 1))

        return scale, dof
    
    def print_labels(self):
        # Count the number of elements in each group
        unique, counts = np.unique(self.labels, return_counts=True)
        group_counts = dict(zip(unique, counts))

        print("Number of elements in each group:")
        for group, count in group_counts.items():
            print(f"Group {group}: {count} elements")