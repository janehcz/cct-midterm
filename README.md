The Cultural Consensus Theory (CCT) model was used to analyze responses from a small dataset on local plant knowledge. Our goal was to estimate the shared cultural knowledge (the consensus answers) and the informant competency levels based on the binary response data. This was done using PyMC to build the model, with the help of AI, and through the commentation of the code to ensure readability and organization of the code. 

Each informant’s competence Dᵢ was given a Uniform(0.5, 1.0) prior, assuming that each person is at least a random chance-level accuracy (50%) and possibly perfect (100%). Dᵢ is a value between 0, wrong answer, and 1, the value that a person gives for a correct answer. For Dᵢ, 0.5 will be pure guessing. This is appropriate because in cultural knowledge contexts, we will assume that the informants are trying their best to answer truthfully with an expectation that they have some basic knowledge or shared cultural knowledge. Consensus answers, Zⱼ, are the shared belief or truth within the culture. This is also a value of 0 (false) or 1 (true). This was modeled with a Bernoulli(0.5) prior as it is uniformative, saying that each answer is equally likely to be 0 or 1 before seeing any of the data. This is appropriate because we will be assuming that there is no prior knowledge of which answer is going to be correct; instead, we let the model learn that from the informant's answers. 

The likelihood function was defined:
    pᵢⱼ = Zⱼ × Dᵢ + (1 - Zⱼ) × (1 - Dᵢ)

This function explains that if someone is competent, they are more likely to give the correct answer. If they are less competent, they might give the right one. For example, if someone is competent with a high Dᵢ, they are more likely to give the correct answer. So if it is 1, they're likely to say 1 and if it's 0, they're likely to say 0. Less competent people are closer to guessing and less helpful for figuring out the consensus.  

The inference was run with 4 chains and 2000 draws per chain. After running, the model converged with no divergences and displayed consistent step sizes as well as equal draws across chains. The posterior distributions were visualized, and the output terminal displayed that the model converged successfully. 

Naive majority voting:
    [0 0 1 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 0 1]

The model’s estimated consensus answer:
    [0 1 1 0 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 1]

The data showed that there were differences at questions [1, 12, 13, 14], which suggests that the model saw something that the majority vote did not. This probably explains that some people were guessing or less reliable on those questions, and it weighted the more competent informants heavily. 

Overall, this shows how CCT can be more accurate than just going with the majority, as it takes into account who is more likely to know what they are talking about.