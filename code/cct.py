import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# load the data
def load_plant_knowledge_data():
    path = "../data/plant_knowledge.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=['Informant'])
    return df.values

# preparing data
data = load_plant_knowledge_data()
N, M = data.shape

# build PyMC model
if __name__ == '__main__':
    with pm.Model() as model:
        # priors
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)  # competence
        Z = pm.Bernoulli("Z", p=0.5, shape=M)  # consensus answers

        # probability of response
        D_reshaped = D[:, None]
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

        # the likelihood
        X = pm.Bernoulli("X", p=p, observed=data)

        # sampling
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True)

    # analysis
    az.summary(trace, var_names=["D", "Z"])

    # posterior of competence
    az.plot_posterior(trace, var_names=["D"], hdi_prob=0.94)
    plt.tight_layout()
    plt.savefig("competence_posteriors.png")
    plt.show()

    # posterior of consensus answers
    az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.94)
    plt.tight_layout()
    plt.savefig("consensus_posteriors.png")
    plt.show()

    # compare with Naive Majority Vote
    majority_vote = data.mean(axis=0) >= 0.5
    posterior_means_Z = trace.posterior["Z"].mean(dim=("chain", "draw")).values
    cct_answers = np.round(posterior_means_Z)

    print("Naive Majority Vote:", majority_vote.astype(int))
    print("CCT Consensus Answers:", cct_answers.astype(int))

    # the differences
    differences = np.where(majority_vote != cct_answers)[0]
    print(f"\nQuestions where CCT and Majority Vote differ: {differences.tolist()}")
