1. In the next portion of today's lecture, we're going to talk about how we can extend policy gradients from the on-policy setting into the off-policy setting.
2. So the first part I want to cover is why policy gradients are considered an on-policy algorithm.
3. Policy gradients are the classical example of an on-policy algorithm because they require generating new samples each time you modify the policy.
4. The reason this is an issue is if you look at the form of the policy gradients, it's an expected value under p theta of τ of ∇log p(τ) times r of tau, and it's really the fact that the expected value is taken under p theta of τ that's the problem.
5. The way that we calculate this expectation in policy gradients is by sampling trajectories using the latest policy.
6. But since the derivative evaluated at parameter vector theta requires samples sampled according to theta, we have to throw out our samples each time we change theta.
7. Which means that policy gradient is an on-policy algorithm.
8. Each update step requires fresh samples.
9. We can't retain data from other policies or even from our own previous policies when using policy gradients.
10. So in the reinforced algorithm, we have step one which is to sample from our policy, step two which is to evaluate the gradient, and step three which is to take a step away in ascent.
11. And we really cannot skip step one.
12. So we can't use samples from past policies, we can't use samples obtained from previous policies, and we can't use samples obtained from previous policies.
13. So we can't use samples from past policies, we can't use samples obtained from previous policies, other sources like demonstrations, we have to generate fresh samples from our own policy every single time.
14. Now this is a bit of a problem when we want to do deep reinforcement learning because neural networks change only a little bit with each gradient step.
15. Because neural networks are highly nonlinear, we can't take really huge gradient steps, which means that in practice we usually end up taking a large number of small gradient steps.
16. But each of those small gradient steps requires generating new samples by running your policy in your system, which might involve actually running your policy in the real world or an expensive simulator.
17. So this can make policy gradients very costly when the cost of generating samples is high, either computational cost or practical monetary cost.
18. So on policy learning can be very inefficient in this way.
19. I should of course mention that on the flip side, if generating samples is very cheap, then policy gradient algorithms can be a great choice because it's a very efficient way to generate samples.
20. So on policy learning, we have to be very careful not to use off-policy samples.
21. They're quite simple, fairly straightforward to implement and tend to work fairly well.
22. But if we do want to use off-policy samples, we can modify policy gradients using something called importance sampling.
23. And that's what we're going to cover next.
24. So what if we don't have samples from P theta of tau?
25. What if we instead have samples from some other distribution that I'm going to call P bar(τ) instead?
26. Now P bar(τ) could be a previous policy gradient, but it's not exactly a positive case policy.
27. So you could be trying to reuse old samples that you've generated, or it could even be some other distribution, like for example, demonstrations from a person.
28. All right, so the trick that we're going to use to modify the policy gradient to accommodate this case is something called importance sampling.
29. Importance sampling is a general technique for evaluating an expectation under one distribution when you only have samples from a different distribution.
30. So here's a sample.
31. Here's how we can write out importance sampling in general.
32. Let's say that we'd like to calculate the expected value of some function f of x under some distribution P of x.
33. We know that the expected value of f of x is the integral over x of P of x times f of x.
34. And if we have access only to some other distribution Q of x, you can multiply the quantity inside the integral by Q of x over Q of x.
35. Right, because you know that Q of x over Q of x is just equal to 1, and you can always multiply by 1 without changing the value.
36. And now we can rearrange these terms a little bit.
37. We can basically say that well Q of x over Q of x times P of x is equal to Q of x times P of x over Q of x.
38. Right, we've just shifted the numerator from one to the other.
39. And now this can be written as an expected value under Q of x.
40. So you can say this is equal to the expected value under Q of X of P of X over Q of X times F of X.
41. There's no approximation here, this is all completely exact, meaning that importance sampling is unbiased.
42. Of course the variance of this estimator could change but in expectation is going to stay the same.
43. So now we're going to apply the same trick to evaluate the policy gradient where the Q here is going to be P bar and the P is going to be P theta.
44. So here is what the importance sample version of the policy gradient of the RL objective would look like.
45. The importance sampled version of the RL objective would be the expected value under some other distribution P bar(τ) of P theta of τ divided by P bar(τ) times R of tau.
46. So that's the RL objective and this is our importance weight.
47. Now if we'd like to understand what the importance weight is equal to, well we can use our identity that describes the trajectory distribution using the chain rule.
48. So we can substitute that in for P theta of τ and P bar of tau.
49. Now we know that both P theta of τ and P bar(τ) have the same initial state distribution P of S1 and the same transition probabilities P of S given S .
50. They only differ by their policy because they both operate in the same MDP.
51. Our distribution has the policy pi theta, the sampling distribution is the policy pi bar.
52. So that means when we take the ratio of the two distributions, the initial state terms and the transition terms cancel and we're just left with a ratio of the products of the policy probabilities.
53. And this is very convenient because in general we don't know P of S1 or P of S given S , but we do know the policy probabilities.
54. So this allows us to actually evaluate these importance weights.
55. Okay, so now let's derive the policy gradient with importance sampling where we're again going to use our convenient identity.
56. So let's say that we have samples from P theta of τ and we want to estimate the value of some new parameter vector theta prime.
57. The objective J(θ) prime will be equal to the expected value under P theta of τ of the importance weight multiplied by the reward.
58. So P theta prime of τ divided by P theta of τ times R of tau.
59. Now notice that here the only part of this objective that actually depends on theta prime that depends on our new parameters is the numerator and the importance weight.
60. Because now our samples are coming from a distribution from a different policy P theta of tau.
61. So that means that when I want to calculate the derivative with respect of theta prime of J(θ) prime, all we have to worry about is this term in the numerator.
62. So this is the derivative.
63. I've just replaced only term that depends on theta prime with its derivative, and then I'm going to substitute my useful identity back in.
64. So the identity tells me that grad theta prime p theta prime of τ is equal to p theta prime of τ times ∇log p theta prime of tau.
65. So I substitute that back in, and I get this equation.
66. Now when you look at this equation, you'll probably immediately recognize it as exactly the equation that we get if we took the policy gradient and just stuck in an importance weight.
67. And in fact, you could derive the importance sample policy gradient that way also.
68. I wanted to derive it in this other way on the slide, just so that you could see the equivalence.
69. Interestingly enough, if you estimate this gradient locally, so if you use this importance sampling derivation to evaluate the gradient at theta equals theta prime, then the importance weight comes out equal to one, and you recover the original policy gradient.
70. So this derivation actually gives you a different way to derive the same policy gradient that we had before.
71. But in the off policy setting, theta prime is not equal to theta, and in that case, we have to fall back on our importance weights, which we derived before, as simply the ratio of the products of the policy probabilities.
72. And if we substituted in all three now the terms in this policy gradient, the importance weights are product overall time steps of pi theta prime over pi theta.
73. The ∇log π part is a sum over all time steps of grad theta prime log π theta prime, and the reward is a sum over all time steps of the reward.
74. So we have three terms inside of our importance sampled off policy policy gradient estimator, and we just multiply those three terms together.
75. Now what about causality?
76. What about the fact that we don't need to consider the effect of current actions on past rewards?
77. Well, we can work those in too, in which case we, again, distribute the terms.
78. So we have three terms inside of our importance sampled off policy gradient estimator, and we just multiply those three terms together.
79. We distribute the rewards and the importance weights into the sum over ∇log pis, and we get a sum from t equals one T of ∇log π times the product of all the importance weights in the past.
80. You can think of that intuitively as the probability that you would have arrived at the state using your new policy, times the sum of rewards weighted by the importance weights in the future.
81. So future actions don't affect the correct weight.
82. That's fine.
83. The trouble is that this last part, you know, this part can be, you know, problematic, can be exponentially large, so can the first part.
84. It turns out that if we ignore this last part, if we ignore the weights on the rewards, we recover something called a policy iteration algorithm.
85. And you can actually prove that a policy iteration algorithm will still improve your policy.
86. It's no longer the gradient, but it's a well-defined way to provide guaranteed improvement to your policy.
87. So don't worry about this yet.
88. We'll cover policy iteration in much more detail in a subsequent lecture.
89. For now, just take my word at it that if you ignore the importance weights that multiply the rewards, if you basically ignore this last term, you still get a procedure that will improve your policy.
90. That is not true for this first term.
91. Okay.
92. So let's look at the first term.
93. The sum, the product from t' equals one to little t of the probability ratios.
94. So this first term is trouble.
95. The reason this first term is trouble is because it's exponential in T.
96. Right.
97. Let's say that the importance weights are all less than one.
98. That's a pretty reasonable assumption because you sampled your actions according to pi theta, so your actions are going to have a higher probability under pi theta than they do under pi theta prime.
99. So, you know, a good chance that your importance weights will be less than one.
100. If you multiply together many, many numbers, each of which is less than one, then their product will go to zero exponentially fast.
101. And that's a really big problem.
102. It essentially means that your variance will go to infinity exponentially fast.
103. And policy gradients already have high variance and now you're going to blow up the variance even more by multiplying them by these high variance importance weights.
104. That's a really bad idea.
105. Now, in order to understand the role that this term plays, we can rewrite our objective a little bit differently.
106. And the reason we're doing all this is because we really just want an excuse to delete that term.
107. So to try to find that excuse, let's write our objective a little bit differently.
108. So here's our on-policy policy gradient.
109. It's a sum over all of our samples, a sum over all of our time steps of ∇log π times this reward to go times this ^{Q}.
110. The ^{Q} is just the sum of all of our time steps.
111. It's a sum from t' equals t to T of the rewards.
112. But I'll write it as ^{Q} because otherwise the notation is going to get pretty hairy.
113. Now, the way that we sampled our SITs and AITs is by actually rolling out our policy in the environment.
114. But you can equivalently think of it as sampling state-action pairs from the state-action marginal at time step t.
115. Because when you sample entire trajectories, the corresponding state-action at every time step, look indistinguishable from what you would have gotten if you sampled from the state-action marginal at that time step.
116. So you could write a different off-policy policy gradient, where instead of importance sampling over entire trajectories, you importance sample over state-action marginals.
117. So now your importance weight is the probability under theta prime of SIT comma AIT, divided by the probability under theta of SIT comma AIT.
118. This is not by itself very useful, because actually, calculating the probabilities for these marginals is impossible without knowledge of the initial state distribution and the transition probabilities.
119. But writing it out in this way allows us to perform a little trick.
120. We can split up using the chain rule, we can split up this marginal, both in the numerator and the denominator, into the product of two terms, a state marginal, pi theta prime of SIT, and the action conditional, pi theta prime of AIT given SIT.
121. And then we could imagine what happens if we just ignore the state marginals, if we just ignore the ratio of the state probabilities.
122. Well, then we get an equation for the importance sampled policy gradient that is very similar to the one I have at the top of the slide, only the product neglects all of the ratios except at t' equals t.
123. So if you don't want your importance weights to be exponential on T, you could try to ignore the ratio, of the state marginal probabilities.
124. So you're still accounting for the ratio of action probabilities, but ignoring the state marginal probabilities.
125. This does not in general give you the correct policy gradient.
126. However, we'll see later on in the course when we discuss advanced policy gradients, that ignoring the state marginal probabilities is reasonable in the sense that it gives you bounded error in the case where theta prime is not too different from theta.
127. And this simple insight is actually very important for deriving practical importance sample policy gradient algorithms that don't suffer from an exponential increase in their variance, right?
128. Because when you multiply together importance weights over all time steps from t' equals 1 to t, you get an exponential increase in variance because your weights exponentially attract to zero.
129. But if you ignore the state marginal rate ratio, then you only get the weights at the time step 2.
130. So you're only going to get a state marginal rate ratio at t, which means that their variance does not grow exponentially.
131. So we'll learn later on when we discuss advanced policy gradients, why ignoring this part is reasonable.
132. For now, I'll just tell you that it's a reasonable choice if theta is close to theta prime, meaning that if your policy is changing only a little bit.