1. Today, we're going to cover our first reinforcement learning algorithm, which is called policy gradient.
2. Now, policy gradients are in some ways kind of the simplest reinforcement learning algorithm, in that they directly attempt to differentiate the reinforcement learning objective, and then perform gradient descent on the policy parameters to make the policy better.
3. So, to start with, let's recap the objective function for reinforcement learning from last time.
4. In reinforcement learning, we have a policy, which we're going to call pi.
5. That policy has parameters, theta, and the policy defines a distribution over actions A, conditioned on either the states S or the observations O.
6. And I'll come back to the partially observed case later in the lecture, but for now, we'll just work on policies that are conditioned on states.
7. If the policy is represented, for example, by a deep neural network, then theta denotes the parameters of the policy, which are the weights.
8. This network takes as input the state or observation, and produces as output the action.
9. Together, the next state is determined by the transition probabilities, which depend on the current state and the action produced by the policy.
10. And, of course, then the next state, sampled according to the transition probabilities, is fed into the policy again to determine the next action, and so on and so on.
11. This process can be used, as we saw, last time to define a trajectory distribution.
12. The trajectory distribution is a probability distribution over a sequence of states and actions.
13. So it's a distribution over S1, A1, S2, A2, S3, A4, A3, S4, etc., etc., etc.
14. And I'm going to use the subscript theta when I write a trajectory distribution to emphasize that the trajectory distribution depends on the policy parameters theta.
15. We can write it via the chain rule of probability, as the product of the initial state distribution, P , and then a product over all time steps of the policy probability, pi theta A , times the transition probability, P , given S .
16. And I will use tau as a notational shorthand.
17. Whenever you see me write tau, that just means S1, A1, S2, A2, S3, etc, etc., etc., all the way out to S , rather than tau .
18. So p , t , sth , t is well<|ru|> hat we observed in this case except for the last two categories.
19. And the last three categories are g , we know both g , but the z , and p , in this case .
20. And t is changeable As we saw in the last lecture, the objective of reinforcement learning can be written out as an expectation under this trajectory distribution.
21. So we have our reward function, R , and we would like to take the expected value of the sum of the reward under the trajectory where the trajectories are distributed according to pθ .
22. And then we would like to find the parameters θ that maximize this expectation.
23. Now as we saw in the last lecture, we can push the sum out of the expectation by linearity of expectation and then express the expectation as an expectation of our marginal.
24. And this allows us to define both a finite horizon version of the R objective and an infinite horizon version.
25. In today's lecture, we will focus on the finite horizon version, although it's quite possible to extend policy gradients to the infinite horizon setting by using value functions.
26. Which we will discuss next time.
27. So for now, we'll stick to the finite horizon version where the sum is inside of the expectation, but we'll come back to the other version later on.
28. Okay, before we talk about how we optimize the reinforcement learning objective, let's first talk about how we can evaluate it.
29. So if we have a policy with parameters θ, can we figure out approximately what is the value of the reinforcement learning objective?
30. And I'm going to use j as a notational shorthand for the expected value under pθ of the sum of the rewards.
31. So if you see me write jθ, I'm just referring to this whole expectation.
32. So if we don't know p and we don't know p given st, how can we estimate j ?
33. So take a moment to think about this.
34. Since we assume that we can run our policy in the real world, which amounts to sampling from the initial state distribution and the transition probabilities, we can evaluate j of θ approximately by simply making rollouts from our policy.
35. We run our policy in the real world n times to collect n sampled trajectories.
36. And if you see me write τ subscript i, that refers to the ith sample.
37. If you see me write s subscript i comma t, that refers to time step t in the ith sample.
38. Having generated these samples from dθ , we can get an unbiased estimate for the expected value of the total reward simply by summing up the rewards along each sample trajectory and then averaging the rewards over the sample trajectory as per this equation.
39. And the more samples we generate, the larger n is, the more accurate will be our estimate of this expected value.
40. So visually you can think of it like this.
41. We will generate some number of trajectories, in this case three, for each trajectory.
42. We'll sum up their rewards to see which ones are good and which ones are bad, and then we'll average them together, and this will give us an estimate of j of theta.
43. Now, of course, in reality, we don't just want to estimate the objective, we actually want to improve it.
44. So to improve the objective, we need to come up with a way to estimate its derivative.
45. And crucially, the estimate of the derivative itself needs to be feasible without knowing the initial state probability nor the transition probability.
46. So again, for notational convenience, I'm going to use p theta of tau to denote the trajectory distribution, and I'll actually use r of tau as shorthand for the sum of the rewards over all the time steps in the trajectory tau.
47. This will make the notation in the derivation that follows a little bit easier to parse.
48. Now, if I have an expected value, I can expand that expected value as a sum for discrete variables or an integral for continuous variables of the product between the probability and the value.
49. So the expected value of r of tau under p theta of tau is equal to the integral over all trajectories of p theta of tau times r of tau.
50. And now we can start working on our derivative.
51. So our goal is to compute the derivative or gradient of j of theta with respect to theta.
52. And since the differentiation operator is linear, we can push it inside the integral.
53. So this derivative is equal to the integral over all trajectories of grad theta p theta tau times r of tau.
54. And I'll often say in this lecture just p of tau.
55. Usually when I say p of tau, I just mean p theta of tau.
56. Okay, so now so far this doesn't actually give us a practical idea of what the derivative of j of theta is.
57. So we're going to use a practical way to evaluate the policy gradient because grad theta p of tau requires differentiating through the unknown initial state distribution and the unknown transition probabilities.
58. But there's a very useful identity that will allow us to rewrite this equation in a way that we can evaluate using only samples, much like how we evaluated the objective value.
59. So the convenient identity that we will use, and this is basically the only piece of mathematical cleverness in this whole derivation, is that if we have an equation like this, if we have p of tau times grad log p of tau, we can write it as p of tau times grad p of tau over p of tau.
60. This follows directly from simply the equation for the derivative of a logarithm.
61. So if you open a calculus textbook and look up the derivative for, you know, d dx of log x, you'll find that it's basically equal to, d x over x, right?
62. So that means that grad log p is grad p over p.
63. But now you'll see that we have a p in the denominator and we have a p in the numerator, so these cancel out, which means that this is equal to grad p.
64. And what we're going to do is we're going to apply this identity in reverse.
65. So we have a grad p here, and we'll substitute the left-hand side of this identity to rewrite it as p times grad log p.
66. And now you'll notice that we have an integral over all trajectories of p of tau times some quantity, which means that we can also write it as an expectation.
67. We can write it as an expected value under p of tau of grad log p tau times r of tau.
68. And this suggests that we might be on the right track, because when we have an expectation, we can evaluate those expectations using samples.
69. But we're not done yet, because we still have this grad log p tau term.
70. So let's work on that a little bit.
71. Let's bring up again our equation for the trajectory distribution.
72. So p of tau, which is just another way of writing p of s1 comma a1 comma s2, etc., is equal to this product that we saw before.
73. If we take the logarithm of both sides, the logarithm of a product is the sum of logarithms, which means that we can write log p of tau as the sum log p of s1 plus a summation from t equals 1 to k.
74. So we can write log p of tau as the sum of the log probabilities under the policy plus the log transition probabilities.
75. And now we'll substitute this whole thing in for grad log p.
76. And we're taking the derivative of this with respect to theta.
77. Now the derivative with respect to theta of log p of s1 is just 0, because p of s1 does not depend on theta.
78. And the derivative with respect to theta of log p of st plus 1 given st at is also 0.
79. Because the transition probabilities also do not depend on theta.
80. So that means that after this simplification, the only terms that are left are the log pi theta at given st terms, which are actually the only terms that we can evaluate, because we know the form of the policy and we can evaluate the policy's own log probabilities.
81. So collecting all the terms that remain and expanding out our notation, we're left with this equation for the policy gradient.
82. The gradient of the equation is the same as the gradient of the equation for the policy gradient.
83. With respect to theta of j of theta is equal to the expectation under p theta of tau of the sum from t equals 1 to capital T of grad theta log pi theta at given st times the sum of the rewards.
84. And now everything inside this expectation is known, because we have access to the policy pi, and we can evaluate the reward for all of our samples.
85. All of the unknown terms, the initial state distribution, and the transition probabilities, occur only in the distribution under which the expectation is taken.
86. So that means that if we want to evaluate the policy gradient, we can use the same trick that we used to evaluate the objective value.
87. We can simply run our policy, which will generate samples from p theta of tau, sum up their rewards to determine which trajectory is good or bad, and then multiply those by the sum of grad log pi's.
88. And then once we've estimated the gradient in this way, we can improve our policy simply by taking a step of gradient descent, taking the old policy parameters and adding to them the policy gradient multiplied by a learning rate alpha.
89. If we think back to the anatomy of a reinforcement learning algorithm that we covered before, the orange box here corresponds to the process of generating those samples, which are the ones that we're summing over.
90. The green box refers to summing up the rewards along each sample trajectory, then we can calculate the policy gradient, and the blue box corresponds to taking one step of gradient descent.
91. Now this procedure gives us the basic policy gradient algorithm, also known as the reinforce algorithm.
92. Reinforce is the acronym that was given by Williams in the 1990s to the first policy gradient method, which consists of three steps.
93. Sample trajectories according to pi theta a given s, by running the policy gradient, and then the simple formula of the formula, which is the first step of gradient descent.
94. So we can take the first step of gradient descent, and then the second step of gradient descent, and then the third step of gradient descent, and then the fourth step of gradient descent, and then the fifth step of gradient descent.
95. So that's the basic policy gradient algorithm.
96. What I've covered so far in this lecture basically gives you all the mathematical tools that you need to understand the basics of policy gradients, but if you try to actually implement the policy gradient, as I've described so far, it probably won't work very well.
97. So in the remainder of the lecture, we'll discuss some of the intuition behind what we're doing, what policy gradients are doing, and then discuss how to actually implement them so that they work well in practice, which you will need to do for homework too.