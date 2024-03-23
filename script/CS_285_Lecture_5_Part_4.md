1. In the next portion of today's lecture, we're going to talk about how we can extend policy gradients from the on-policy setting into the off-policy setting.
2. So the first part I want to cover is why policy gradients are considered an on-policy algorithm.
3. Policy gradients are the classical example of an on-policy algorithm because they require generating new samples each time you modify the policy.
4. The reason this is an issue is if you look at the form of the policy gradients, it's an expected value under p_θ(τ) of ∇log p(τ) times r(τ), and it's really the fact that the expected value is taken under p_θ(τ) that's the problem.
5. The way that we calculate this expectation in policy gradients is by sampling trajectories using the latest policy.
6. But since the derivative evaluated at parameter vector θ requires samples sampled according to θ, we have to throw out our samples each time we change θ.
7. Which means that policy gradient is an on-policy algorithm.
8. Each update step requires fresh samples.
9. We can't retain data from other policies or even from our own previous policies when using policy gradients.
10. So in the REINFORCE algorithm, we have step one which is to sample from our policy, step two which is to evaluate the gradient, and step three which is to take a step gradient ascent.
11. And we really cannot skip step one.
12. So we can't use samples from past policies, we can't use samples obtained from other sources like demonstrations, we have to generate fresh samples from our own policy every single time.
13. Now this is a bit of a problem when we want to do deep reinforcement learning because neural networks change only a little bit with each gradient step.
14. Because neural networks are highly nonlinear, we can't take really huge gradient steps, which means that in practice we usually end up taking a large number of small gradient steps.
15. But each of those small gradient steps requires generating new samples by running your policy in your system, which might involve actually running your policy in the real world or an expensive simulator.
16. So this can make policy gradients very costly when the cost of generating samples is high, either computational cost or practical monetary cost.
17. So on policy learning can be very inefficient in this way.
18. I should of course mention that on the flip side, if generating samples is very cheap, then policy gradient algorithms can be a great choice because they're quite simple, fairly straightforward to implement and tend to work fairly well
19. But if we do want to use off-policy samples, we can modify policy gradients using something called importance sampling.
20. And that's what we're going to cover next.
21. So what if we don't have samples from p_θ(τ)?
22. What if we instead have samples from some other distribution that I'm going to call bar{p}(τ) instead?
23. Now bar{p}(τ) could be a previous policy you could be trying to reuse old samples that you've generated, or it could even be some other distribution, like for example, demonstrations from a person.
24. All right, so the trick that we're going to use to modify the policy gradient to accommodate this case is something called importance sampling.
25. Importance sampling is a general technique for evaluating an expectation under one distribution when you only have samples from a different distribution.
26. So here's a sample.
27. Here's how we can write out importance sampling in general.
28. Let's say that we'd like to calculate the expected value of some function f(x) under some distribution p(x).
29. We know that the expected value of f(x) is the integral over x of p(x) times f(x).
30. And if we have access only to some other distribution q(x), you can multiply the quantity inside the integral by q(x)/q(x).
31. Right, because you know that q(x)/q(x) is just equal to 1, and you can always multiply by 1 without changing the value.
32. And now we can rearrange these terms a little bit.
33. We can basically say that well q(x)/q(x) times p(x) is equal to q(x) times p(x)/q(x).
34. Right, we've just shifted the numerator from one to the other.
35. And now this can be written as an expected value under q(x).
36. So you can say this is equal to the expected value under q(x) of p(x)/q(x) times f(x).
37. There's no approximation here, this is all completely exact, meaning that importance sampling is unbiased.
38. Of course the variance of this estimator could change but in expectation is going to stay the same.
39. So now we're going to apply the same trick to evaluate the policy gradient where the q here is going to be bar{p} and the p is going to be p_θ.
40. So here is what the importance sample version of the policy gradient of the RL objective would look like.
41. The importance sampled version of the RL objective would be the expected value under some other distribution bar{p}(τ) of p_θ(τ)/bar{p}(τ) times r(τ).
42. So that's the RL objective and this is our importance weight.
43. Now if we'd like to understand what the importance weight is equal to, well we can use our identity that describes the trajectory distribution using the chain rule.
44. So we can substitute that in for p_θ(τ) and bar{p}(τ).
45. Now we know that both p_θ(τ) and bar{p}(τ) have the same initial state distribution p(s_1) and the same transition probabilities p(s_{t+1}|s_t,a_t).
46. They only differ by their policy because they both operate in the same MDP.
47. Our distribution has the policy π_θ, the sampling distribution is the policy bar{π}.
48. So that means when we take the ratio of the trajectory distributions, the initial state terms and the transition terms cancel and we're just left with a ratio of the products of the policy probabilities.
49. And this is very convenient because in general we don't know p(s_1) or p(s_{t+1}|s_t,a_t), but we do know the policy probabilities.
50. So this allows us to actually evaluate these importance weights.
51. Okay, so now let's derive the policy gradient with importance sampling where we're again going to use our convenient identity.
52. So let's say that we have samples from p_θ(τ) and we want to estimate the value of some new parameter vector θ'.
53. The objective J(θ) prime will be equal to the expected value under p_θ(τ) of the importance weight multiplied by the reward.
54. So p_{θ'}(τ)/p_θ(τ) times r(τ).
55. Now notice that here the only part of this objective that actually depends on θ' that depends on our new parameters is the numerator and the importance weight.
56. Because now our samples are coming from a distribution from a different policy p_θ(τ).
57. So that means that when I want to calculate the derivative with respect of θ' of J(θ) prime, all we have to worry about is this term in the numerator.
58. So this is the derivative.
59. I've just replaced only term that depends on θ' with its derivative, and then I'm going to substitute my useful identity back in.
60. So the identity tells me that ∇_{θ'} p_{θ'}(τ) is equal to p_{θ'}(τ) times ∇log p_{θ'}(τ).
61. So I substitute that back in, and I get this equation.
62. Now when you look at this equation, you'll probably immediately recognize it as exactly the equation that we get if we took the policy gradient and just stuck in an importance weight.
63. And in fact, you could derive the importance sample policy gradient that way also.
64. I wanted to derive it in this other way on the slide, just so that you could see the equivalence.
65. Interestingly enough, if you estimate this gradient locally, so if you use this importance sampling derivation to evaluate the gradient at θ = θ', then the importance weight comes out equal to one, and you recover the original policy gradient.
66. So this derivation actually gives you a different way to derive the same policy gradient that we had before.
67. But in the off policy setting, θ' is not equal to θ, and in that case, we have to fall back on our importance weights, which we derived before, as simply the ratio of the products of the policy probabilities.
68. And if we substituted in all three now the terms in this policy gradient, the importance weights are product overall time steps of π_{θ'}/π_θ.
69. The ∇log π part is a sum over all time steps of ∇_{θ'} log π_{θ'}, and the reward is a sum over all time steps of the reward.
70. So we have three terms inside of our importance sampled off policy policy gradient estimator, and we just multiply those three terms together.
71. Now what about causality?
72. What about the fact that we don't need to consider the effect of current actions on past rewards?
73. Well, we can work those in too, in which case we, again, distribute the rewards and the importance weights into the sum over ∇log π's, and we get a sum from t=1 T of ∇log π times the product of all the importance weights in the past.
74. You can think of that intuitively as the probability that you would have arrived at the state using your new policy, times the sum of rewards weighted by the importance weights in the future.
75. So future actions don't affect the correct weight.
76. That's fine.
77. The trouble is that this last part, you know, this part can be, you know, problematic, can be exponentially large, so can the first part.
78. It turns out that if we ignore this last part, if we ignore the weights on the rewards, we recover something called a policy iteration algorithm.
79. And you can actually prove that a policy iteration algorithm will still improve your policy.
80. It's no longer the gradient, but it's a well-defined way to provide guaranteed improvement to your policy.
81. So don't worry about this yet.
82. We'll cover policy iteration in much more detail in a subsequent lecture.
83. For now, just take my word at it that if you ignore the importance weights that multiply the rewards, if you basically ignore this last term, you still get a procedure that will improve your policy.
84. That is not true for this first term.
85. The sum, the product from t'=1 to t of the probability ratios.
86. So this first term is trouble.
87. The reason this first term is trouble is because it's exponential in T.
88. Right.
89. Let's say that the importance weights are all less than one.
90. That's a pretty reasonable assumption because you sampled your actions according to π_θ, so your actions are going to have a higher probability under π_θ than they do under π_{θ'}.
91. So, you know, a good chance that your importance weights will be less than one.
92. If you multiply together many, many numbers, each of which is less than one, then their product will go to zero exponentially fast.
93. And that's a really big problem.
94. It essentially means that your variance will go to infinity exponentially fast.
95. And policy gradients already have high variance and now you're going to blow up the variance even more by multiplying them by these high variance importance weights.
96. That's a really bad idea.
97. Now, in order to understand the role that this term plays, we can rewrite our objective a little bit differently.
98. And the reason we're doing all this is because we really just want an excuse to delete that term.
99. So to try to find that excuse, let's write our objective a little bit differently.
100. So here's our on-policy policy gradient.
101. It's a sum over all of our samples, a sum over all of our time steps of ∇log π times this reward to go times this ^{Q}.
102. The ^{Q} is just the sum from t' equals t to T of the rewards.
103. But I'll write it as ^{Q} because the notation is going to get pretty hairy.
104. Now, the way that we sampled our s_{i,t}s and a_{i,t}s is by actually rolling out our policy in the environment.
105. But you can equivalently think of it as sampling state-action pairs from the state-action marginal at time step t.
106. Because when you sample entire trajectories, the corresponding state-action at every time step, look indistinguishable from what you would have gotten if you sampled from the state-action marginal at that time step.
107. So you could write a different off-policy policy gradient, where instead of importance sampling over entire trajectories, you importance sample over state-action marginals.
108. So now your importance weight is the probability under θ'(s_{i,t},a_{i,t}), divided by the probability under θ(s_{i,t},a_{i,t}).
109. This is not by itself very useful, because actually, calculating the probabilities for these marginals is impossible without knowledge of the initial state distribution and the transition probabilities.
110. But writing it out in this way allows us to perform a little trick.
111. We can split up using the chain rule, we can split up this marginal, both in the numerator and the denominator, into the product of two terms, a state marginal, π_{θ'}(s_{i,t}), and the action conditional, π_{θ'}(a_{i,t}|s_{i,t}).
112. And then we could imagine what happens if we just ignore the state marginals, if we just ignore the ratio of the state probabilities.
113. Well, then we get an equation for the importance sampled policy gradient that is very similar to the one I have at the top of the slide, only the product neglects all of the ratios except at t' equals t.
114. So if you don't want your importance weights to be exponential on T, you could try to ignore the ratio, of the state marginal probabilities.
115. So you're still accounting for the ratio of action probabilities, but ignoring the state marginal probabilities.
116. This does not in general give you the correct policy gradient.
117. However, we'll see later on in the course when we discuss advanced policy gradients, that ignoring the state marginal probabilities is reasonable in the sense that it gives you bounded error in the case where θ' is not too different from θ.
118. And this simple insight is actually very important for deriving practical importance sample policy gradient algorithms that don't suffer from an exponential increase in their variance, right?
119. Because when you multiply together importance weights over all time steps from t'=1 to t, you get an exponential increase in variance because your weights exponentially attract to zero.
120. But if you ignore the state marginal rate ratio, then you only get the weights at the time step t, which means that their variance does not grow exponentially.
121. So we'll learn later on when we discuss advanced policy gradients, why ignoring this part is reasonable.
122. For now, I'll just tell you that it's a reasonable choice if θ is close to θ', meaning that if your policy is changing only a little bit.
