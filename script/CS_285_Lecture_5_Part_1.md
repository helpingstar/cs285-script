1. Today, we're going to cover our first reinforcement learning algorithm, which is called policy gradient.
2. Now, policy gradients are in some ways kind of the simplest reinforcement learning algorithm, in that they directly attempt to differentiate the reinforcement learning objective, and then perform gradient descent on the policy parameters to make the policy better.
3. So, to start with, let's recap the objective function for reinforcement learning from last time.
4. In reinforcement learning, we have a policy, which we're going to call π.
5. That policy has parameters, θ, and the policy defines a distribution over actions A, conditioned on either the states S or the observations O.
6. And I'll come back to the partially observed case later in the lecture, but for now, we'll just work on policies that are conditioned on states.
7. If the policy is represented, for example, by a deep neural network, then θ denotes the parameters of the policy, which are the weights in the neural network.
8. This network takes as input the state or observation, and produces as output the action.
9. Together, the next state is determined by the transition probabilities, which depend on the current state and the action produced by the policy.
10. And, of course, then the next state, sampled according to the transition probabilities, is fed into the policy again to determine the next action, and so on and so on.
11. This process can be used, as we saw, last time to define a trajectory distribution.
12. The trajectory distribution is a probability distribution over a sequence of states and actions.
13. So it's a distribution over s_1, a_1, s_2, a_2, s_3, a_4, a_3, s_4, etc., etc., etc.
14. And I'm going to use the subscript θ when I write a trajectory distribution to emphasize that the trajectory distribution depends on the policy parameters θ.
15. We can write it via the chain rule of probability, as the product of the initial state distribution, p(s_1) , and then a product over all time steps of the policy probability, π_θ(a_t|s_T), times the transition probability, p(s_{t+1}).
16. And I will use τ(tau) as a notational shorthand.
17. Whenever you see me write τ, that just means s_1, a_1, s_2, a_2, s_3, etc, etc., etc., all the way out to s_T, a_T.
18. Now, crucially when we develop model free reinforcement learning algorithms of the sort that we'll cover in today's lecture and the subsequent few lectures we typically do not assume that we know the transition probabilities p(s_{t+1}|s_t, a_t) nor the initial state probability p(s_1).
19. We just assume that we can interact with the real world which effectively samples from those distributions.
20. As we saw in the last lecture, the objective of reinforcement learning can be written out as an expectation under this trajectory distribution.
21. So we have our reward function, r(s_t, a_t), and we would like to take the expected value of the sum of the reward under the trajectory where the trajectories are distributed according to p_θ(τ).
22. And then we would like to find the parameters θ that maximize this expectation.
23. Now as we saw in the last lecture, we can push the sum out of the expectation by linearity of expectation and then express the expectation as an expectation of our marginal.
24. And this allows us to define both a finite horizon version of the R objective and an infinite horizon version.
25. In today's lecture, we will focus on the finite horizon version, although it's quite possible to extend policy gradients to the infinite horizon setting by using value functions.
26. Which we will discuss next time.
27. So for now, we'll stick to the finite horizon version where the sum is inside of the expectation, but we'll come back to the other version later on.
28. Okay, before we talk about how we optimize the reinforcement learning objective, let's first talk about how we can evaluate it.
29. So if we have a policy with parameters θ, can we figure out approximately what is the value of the reinforcement learning objective?
30. And I'm going to use J(θ) as a notational shorthand for the expected value under p_θ(τ) of the sum of the rewards.
31. So if you see me write J(θ), I'm just referring to this whole expectation.
32. So if we don't know p(s_1) and we don't know p(s_{t+1}|s_t), how can we estimate J(θ)?
33. So take a moment to think about this.
34. Since we assume that we can run our policy in the real world, which amounts to sampling from the initial state distribution and the transition probabilities, we can evaluate J(θ) approximately by simply making rollouts from our policy.
35. We run our policy in the real world N times to collect N sampled trajectories.
36. And if you see me write τ_i, that refers to the ith sample.
37. If you see me write s_{i,t}, that refers to time step t in the ith sample.
38. Having generated these samples from p_θ(τ), we can get an unbiased estimate for the expected value of the total reward simply by summing up the rewards along each sample trajectory and then averaging the rewards over the sample trajectory as per this equation.
39. And the more samples we generate, the larger N is, the more accurate will be our estimate of this expected value.
40. So visually you can think of it like this.
41. We will generate some number of trajectories, in this case three, for each trajectory.
42. We'll sum up their rewards to see which ones are good and which ones are bad, and then we'll average them together, and this will give us an estimate of J(θ).
43. Now, of course, in reality, we don't just want to estimate the objective, we actually want to improve it.
44. So to improve the objective, we need to come up with a way to estimate its derivative.
45. And crucially, the estimate of the derivative itself needs to be feasible without knowing the initial state probability nor the transition probability.
46. So again, for notational convenience, I'm going to use p_θ(τ) to denote the trajectory distribution, and I'll actually use r(τ) as shorthand for the sum of the rewards over all the time steps in the trajectory τ.
47. This will make the notation in the derivation that follows a little bit easier to parse.
48. Now, if I have an expected value, I can expand that expected value as a sum for discrete variables or an integral for continuous variables of the product between the probability and the value.
49. So the expected value of r(τ) under p_θ(τ) is equal to the integral over all trajectories of p_θ(τ) times r(τ).
50. And now we can start working on our derivative.
51. So our goal is to compute the derivative or gradient of J(θ) with respect to θ.
52. And since the differentiation operator is linear, we can push it inside the integral.
53. So this derivative is equal to the integral over all trajectories of ∇_θ p_θ(τ) times r(τ).
54. And I'll often say in this lecture just p(τ).
55. Usually when I say p(τ), I just mean p_θ(τ).
56. Okay, so now so far this doesn't actually give us a practical way to evaulate the policy gradient because ∇_θ p_θ(τ) requires differentiating the unknown initial state distribution and the unknown transition probabilities but there's a very useful identity that will allow us to rewrite this equation in a way that we can evaluate using only samples, much like how we evaluated the objective value.
57. So the convenient identity that we will use, and this is basically the only piece of mathematical cleverness in this whole derivation, is that if we have an equation like this, if we have p(τ) times ∇log p(τ), we can write it as p(τ) times ∇p(τ) over p(τ).
58. This follows directly from simply the equation for the derivative of a logarithm.
59. So if you open a calculus textbook and look up the derivative for, you know, d dx of log x, you'll find that it's basically equal to, d x over x, right?
60. So that means that ∇log p is ∇p over p.
61. But now you'll see that we have a p in the denominator and we have a p in the numerator, so these cancel out, which means that this is equal to ∇p.
62. And what we're going to do is we're going to apply this identity in reverse.
63. So we have a ∇p here, and we'll substitute the left-hand side of this identity to rewrite it as p times ∇log p.
64. And now you'll notice that we have an integral over all trajectories of p(τ) times some quantity, which means that we can also write it as an expectation.
65. We can write it as an expected value under p(τ) of ∇log p(τ) times r(τ).
66. And this suggests that we might be on the right track, because when we have an expectation, we can evaluate those expectations using samples.
67. But we're not done yet, because we still have this ∇log p(τ) term.
68. So let's work on that a little bit.
69. Let's bring up again our equation for the trajectory distribution.
70. So p(τ), which is just another way of writing p(s_1,a_1,s_2,...) is equal to this product that we saw before.
71. If we take the logarithm of both sides, the logarithm of a product is the sum of logarithms, which means that we can write log p(τ) as the sum log p(s_1) plus a summation from t equals 1 to T of the log probabilities under the policy plus the log transition probabilities.
73. And now we'll substitute this whole thing in for ∇log p.
74. And we're taking the derivative of this with respect to θ.
75. Now the derivative with respect to θ of log p(s_1) is just 0, because p(s_1) does not depend on θ.
76. And the derivative with respect to θ of log p(s_{t+1}|s_t,a_t) is also 0.
77. Because the transition probabilities also do not depend on θ.
78. So that means that after this simplification, the only terms that are left are the log π_θ(a_t|s_t) terms, which are actually the only terms that we can evaluate, because we know the form of the policy and we can evaluate the policy's own log probabilities.
79. So collecting all the terms that remain and expanding out our notation, we're left with this equation for the policy gradient.
80. The gradient with respect to the θ J(θ) is equal to the expectation under p_θ(τ) of the sum from t=1 to T of ∇_θ log π_θ(a_t|s_t) times the sum of the rewards.
81. And now everything inside this expectation is known, because we have access to the policy π, and we can evaluate the reward for all of our samples.
82. All of the unknown terms, the initial state distribution, and the transition probabilities, occur only in the distribution under which the expectation is taken.
83. So that means that if we want to evaluate the policy gradient, we can use the same trick that we used to evaluate the objective value.
84. We can simply run our policy, which will generate samples from p_θ(τ), sum up their rewards to determine which trajectory is good or bad, and then multiply those by the sum of ∇log π's.
85. And then once we've estimated the gradient in this way, we can improve our policy simply by taking a step of gradient descent, taking the old policy parameters and adding to them the policy gradient multiplied by a learning rate α.
86. If we think back to the anatomy of a reinforcement learning algorithm that we covered before, the orange box here corresponds to the process of generating those samples, which are the ones that we're summing over.
87. The green box refers to summing up the rewards along each sample trajectory, then we can calculate the policy gradient, and the blue box corresponds to taking one step of gradient descent.
88. Now this procedure gives us the basic policy gradient algorithm, also known as the REINFORCE algorithm.
89. REINFORCE is the acronym that was given by Williams in the 1990s to the first policy gradient method, which consists of three steps.
90. Sample trajectories according to π_θ(a|s), by running the policy in the real world N times evaluate the policy gradient as for this equation, and then take a stpe of gradient descent.
91. So that's the basic policy gradient algorithm.
92. What I've covered so far in this lecture basically gives you all the mathematical tools that you need to understand the basics of policy gradients, but if you try to actually implement the policy gradient, as I've described so far, it probably won't work very well.
93. So in the remainder of the lecture, we'll discuss some of the intuition behind what policy gradients are doing, and then discuss how to actually implement them so that they work well in practice, which you will need to do for homework too.