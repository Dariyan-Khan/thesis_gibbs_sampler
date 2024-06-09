from scipy.linalg import orthogonal_procrustes
import numpy as np
from scipy.stats import beta
from scipy.stats import multivariate_normal, truncnorm, invwishart


class Synthetic_data():

    def __init__(self, μ_1, μ_2, K, prior, N_t=1000):
        # prior has to be a function which takes no arguments and spits out a sample

        # assert len(μ_1) == 2, "μ_1 and μ_2 have to be a 2D vector"
        # assert len(μ_1) == len(μ_2), "μ_1 and μ_2 have to be a 2D vector"

        self.μ_1 = μ_1
        self.μ_2 = μ_2
        self.K = K
        self.mean_len = len(μ_1)
        self.N_t=N_t
        self.adj_mat, self.bern_params = self.simulate_adj_mat(prior, μ_1, μ_2)
        self.embds = self.spectral_emb()
        self.normed_embds = self.embds / np.linalg.norm(self.embds, axis=1)[:, np.newaxis]


        self.bern_params = np.array(self.bern_params)
        self.true_labels = self.bern_params[:,1]
        self.true_labels = np.array(self.true_labels, dtype=int)

        self.angles = [np.arctan2(x, y) for x,y in self.embds]

        self.mean_samples = []
        self.Sigma_samples = []
        self.r_samples = []
        self.z_samples = []
        self.pi_samples = []

        self.prior_df = 2
        self.prior_scale_mat = np.eye(self.mean_len)

        self.pi_prior = 1.0 / self.K

    def find_delta_inv(self, μ_1, μ_2, exp_rho):
        μ_1_outer = np.outer(μ_1, μ_1)
        μ_2_outer = np.outer(μ_2, μ_2)

        Δ = exp_rho**2 * (1 / 2) * (μ_1_outer + μ_2_outer)

        Δ_inv = np.linalg.inv(Δ) 
        return Δ_inv

    def exp_X1_inner_func(self, x, ρ, μ):
        return (np.dot(x, ρ*μ) - (np.dot(x, ρ*μ)**2)) * np.outer(ρ*μ, ρ*μ)

    def covariance_estimate(self, x, μ_1, μ_2, prior, exp_rho, N_ρ=1000, N_t=1000):
        ρ_samples_1 = np.array([prior() for _ in range(N_ρ)])
        ρ_samples_2 = np.array([prior() for _ in range(N_ρ)])
        μ_1_integral_estimate = (1 / N_ρ) * sum(self.exp_X1_inner_func(x, ρ, μ_1) for ρ in ρ_samples_1)
        μ_2_integral_estimate = (1 / N_ρ) * sum(self.exp_X1_inner_func(x, ρ, μ_2) for ρ in ρ_samples_2)
        exp_X1_func_estimate = 0.5 * (μ_1_integral_estimate + μ_2_integral_estimate)
        Δ_inv = self.find_delta_inv(μ_1, μ_2, exp_rho)
        return (Δ_inv @ exp_X1_func_estimate @ Δ_inv) / N_t

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def simulate_adj_mat(self, prior, μ_1, μ_2):
        μ_mat = np.stack((μ_1, μ_2), axis=1)
        bern_params = [(prior(), np.random.randint(0,2)) for _ in range(self.N_t)]
        # bern_params = [(prior(), i % 2) for i in range(self.N_t)]
        ## np.random.shuffle(bern_params) # shuffle the bernoulli parameters for testing purposes
        adj_mat = np.zeros((self.N_t, self.N_t))

        for i in range(self.N_t):
            ρ_i, μ_i = bern_params[i][0], μ_mat[:, bern_params[i][1]]
            for j in range(i):
                ρ_j, μ_j = bern_params[j][0], μ_mat[:, bern_params[j][1]]

                adj_mat[i,j] = np.random.binomial(1, ρ_i * ρ_j * np.dot(μ_i, μ_j))

                adj_mat[j,i] = adj_mat[i,j]
            
            adj_mat[i,i] = 1
        
        assert self.check_symmetric(adj_mat)

        return adj_mat, bern_params
    
    def spectral_emb(self):
        μ_mat = np.stack((self.μ_1, self.μ_2), axis=1)
        eigvals, eigvecs = np.linalg.eig(self.adj_mat)
        sorted_indexes = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[sorted_indexes]
        eigvecs = eigvecs[:,sorted_indexes]
        embedding_dim = self.d # should be 2
        eigvecs_trunc = eigvecs[:,:embedding_dim]
        eigvals_trunc = np.diag(np.sqrt(np.abs(eigvals[:embedding_dim])))
        spectral_embedding = eigvecs_trunc @ eigvals_trunc
        true_means = np.zeros((self.N_t, self.mean_len))

        for i in range(self.N_t):
            ρ_i, μ_i = self.bern_params[i][0], μ_mat[:, self.bern_params[i][1]]
            true_means[i, :] =  ρ_i * μ_i

        # print("true_means", true_means)

        best_orthog_mat = orthogonal_procrustes(spectral_embedding, true_means)

        spectral_embedding = spectral_embedding @ best_orthog_mat[0]

        # print("spectral_embedding", spectral_embedding)

        # print("min norm", min(spectral_embedding, key=np.linalg.norm))

        return spectral_embedding
    
    def Sigma_prior(self, x):
        return invwishart.pdf(x, self.prior_df, scale=self.prior_scale_mat)

    
    def calculate_density(self, i, mu_zi, Sigma_zi):
        """
        Calculate the probability density of the vector tilde{x}    r_i for a multivariate normal distribution
        characterized by mean mu_zi and covariance Sigma_zi.
        """
        curr_r_i = self.r_samples[-1][i]
        d = len(mu_zi)
        density = curr_r_i **(d-1) * multivariate_normal.pdf(self.embds[i], mean=mu_zi, cov=Sigma_zi)
        return density
        


    def z_update(self, i):
        z_probs = [] # list to contain z_probs that we then sample from
        for k in range(self.K):
            z_probs = self.pi_samples[-1][k] * self.calculate_density(i, self.mean_samples[-1][k], self.Sigma_samples[-1][k])
        
        z_probs = np.array(z_probs)
        z_probs = z_probs / np.sum(z_probs)
        z_i = np.random.choice(self.K, p=z_probs)
        return z_i
    
    
    def sample_r_i(self, i, mu_zi, Sigma_zi):
        """
        Generate a sample for r_i using a truncated normal distribution centered at mu_zi
        with the covariance matrix Sigma_zi as a scale parameter.
        """
        a, b = 0, np.inf  # Truncation limits
        scale = np.sqrt(np.diag(Sigma_zi))  # Scale is the standard deviation
        curr_r_i = self.r_samples[-1][i]
        lower, upper = (a - mu_zi) / scale, (b - mu_zi) / scale
        proposal_r_i = truncnorm.rvs(lower, upper, loc=mu_zi, scale=scale, size=len(mu_zi))
        
        # Calculate the densities
        current_density = self.calculate_density(i, mu_zi, Sigma_zi)
        proposal_density = self.calculate_density(i, proposal_r_i, Sigma_zi)
        
        # Calculate acceptance probability
        acceptance_probability = min(1, proposal_density / current_density)
        
        # Accept or reject the proposal
        if np.random.rand() < acceptance_probability:
            return proposal_r_i
        else:
            return curr_r_i
    
    
    def mean_and_sigma_sample_density(self, μ_k, Σ_k, k):
        prob = 1
        for i, group in enumerate(self.z_samples[-1]):
            if group == k:
                prob *= self.calculate_density(i, μ_k, Σ_k)
        
        prob * multivariate_normal.pdf(self.embds[i], mean=μ_k, cov=Σ_k)
        prob *= self.Sigma_prior(Σ_k)
        return prob
    

    
    def sample_mean_and_sigma(self, k):
        # Current parameters
        current_mu_k = self.mu[-1][k]
        current_sigma_k = self.sigma[-1][k]
        
        # Proposal generation
        proposed_mu_k = multivariate_normal.rvs(mean=current_mu_k, cov=np.eye(len(current_mu_k)) * 0.01)
        proposed_sigma_k = invwishart.rvs(df=len(current_sigma_k) + 1, scale=current_sigma_k)
        
        # Compute the densities for the current and proposed parameters
        current_density = self.sample_density(current_mu_k, current_sigma_k, k)
        proposed_density = self.sample_density(proposed_mu_k, proposed_sigma_k, k)
        
        # Compute acceptance probability
        acceptance_ratio = proposed_density / current_density
        
        # Accept or reject the proposal based on the computed probability
        if np.random.rand() < acceptance_ratio:
            return proposed_mu_k, proposed_sigma_k
        else:
            return current_mu_k, current_sigma_k
    


    
    def sample_pi(self):
        num_in_each_group = np.zeros(self.K)
        for group in self.z_samples[-1]:
            num_in_each_group[group] += 1
        
        # Sample from the Dirichlet distribution with parameters (n_1 + self.pi_prior, ..., n_K + self.pi_prior)

        new_pi = np.random.dirichlet(num_in_each_group + self.pi_prior)

        return new_pi
    


    def gibbs_sampler(self):

        





if __name__ == '__main__':

    μ_1 = np.array([0.75, 0.25])
    μ_2 = np.array([0.25, 0.75])

    α = 2
    β = 2
    prior = lambda : beta.rvs(α, β)
    # prior = lambda : 1

    ds = Synthetic_data(μ_1, μ_2, prior, N_t=1000)

