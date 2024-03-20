1. Alright, now that we've covered the mathematical derivation for policy gradients, let's work a little bit on developing some intuition for what policy gradients are actually doing.
2. Alright, so these are the equations that we saw before.
3. We've got the approximate expression for the derivative of j theta, which is a sum over all of our samples of the sum of grad log pi's along that sample trajectory times the total reward of that trajectory.
4. So what is this grad log pi thing actually?
5. Well, let's say that our policy for now is just a discrete.
6. Let's say that it's just a mapping from images, maybe these are driving images, to a discrete action turn left or turn right.
7. Then log pi is simply the log probability that this policy assigns to one of those two actions.
8. And grad log pi is the derivative of that log probability.
9. So a neural network will output those probabilities and you can take the logarithm of that probability.
10. When you do maximum likelihood training supervised learning, you're typically maximizing log probabilities of observed labels.
11. So it's instructive, perhaps, to compare what policy gradients are doing to what maximum likelihood is doing.
12. So in maximum likelihood, like in imitation learning, for instance, we would collect some data of humans selecting actions and then we would run supervised learning on that data and that would yield a policy pi whate.
13. The maximum likelihood objective or the supervised learning objective is just maximization of the log probabilities assigned to the observed actions.
14. So the gradient of that is given by the sum over all of your samples and all your time steps of grad log pi A i t given st.
15. So the gradient of that is given by the sum over all of your samples and all your time of grad log pi A given S .
16. Now of course when we're doing maximum likelihood, we assume that the actions in our data, A , are good actions to take.
17. In policy gradient that is not necessarily true because we generated those actions by running our own previous policy, which might not have been very good.
18. So the maximum likelihood gradient simply increases the log probabilities of all the actions, whereas the policy gradient might increase or decrease them depending on the value of their reward.
19. So intuitively high reward trajectories get their log probabilities increased, low reward trajectories get their log probabilities decreased.
20. So you can think of it as a kind of weighted version of the gradient for the maximum likelihood objective.
21. In fact this interpretation will turn out to be very useful when it comes time to actually implement the policy gradient with modern automatic differentiation tools like PyTorch.
22. Now that was an example with discrete actions.
23. What if we have continuous actions?
24. What if for example we want to make this little humanoid robot run using policy gradients?
25. Well in that case we need to select a representation for pi that can output distributions over continuous valued actions.
26. For example we might represent pi theta A given S as a multivariate normal distribution or Gaussian distribution where the mean is given by a neural network.
27. So the neural network outputs the mean and then the mean outputs the value.
28. Then you have some variance which could be learned or could be fixed and then you would like to train this neural network.
29. In that case you can write the log probability by using the formula for the log probability under a multivariate normal distribution, which is simply the difference between the mean and the action under the covariance, the inverse covariance matrix.
30. So this is one way of writing the log probability of a multivariate normal distribution.
31. And you can then calculate the derivative of this thing with respect to the the mean and you just get this equation.
32. So the derivative of your multivariate normal is just negative one half times the inverse covariance times fst minus at times df d theta.
33. And in practice the way that you would calculate this quantity is you would compute negative one half sigma inverse f of s d minus at and then back propagate it through your network to get the derivative with respect to theta.
34. So that's the way you would write the gradient of the gradient of the gradient.
35. So that's the way you would write the gradient of the gradient.
36. All right so that maybe gives us some intuition for what these grad log pi terms are actually doing both in the discrete action and continuous action case.
37. In both cases they correspond to a kind of weighted version of the maximum likelihood gradient, if it's helpful for you to think about it that way.
38. And you can compute them by basically using the formula for the log probability of whatever distribution class you choose to use.
39. So I'll collect some of the terms and and use slightly more concise notation to make this a little clearer so you can equivalently write it as grad log pi theta of tau times r of tau where this grad log pi theta tau is just the sum over the individual grad log pi thetas.
40. The maximum likelihood gradient is given here so it's just the same thing only without the r term.
41. So intuitively what that means is that if you roll out some trajectories and you compute their rewards and some of them have big positive rewards represented with green check marks and some of them have big negative rewards represented by the red x and some are kind of neutral like that middle one what you'd like to do is you'd like to take the log probabilities along the good trajectories and raise them and take the log probabilities along the bad trajectories and lower them.
42. So the policy gradient makes the good stuff more likely, and makes the bad stuff less likely.
43. So in a sense you can think of the policy gradient as a kind of formalization of trial and error learning.
44. If reinforcement learning refers to learning about trial and error then policy gradient simply formalizes that notion as a gradient ascent algorithm.
45. Now what I would like to briefly mention next is a short aside regarding partial observability.
46. So if we want to learn policy gradient, we need to first understand already the vs and aparecer graduation rates.
47. By creating an original classification for not with rails, we're trying to avoid Millier's Moving Lines.
48. And So that means that we initially stick to the the three fixed bias to achieve ascentosta� Franklin, We's got to think the Sh 되어 s at the res the normal tension that is being captured.
49. If the rlo is greater than different values in between the세요 this is the time.
50. which means that if you wanted to derive the policy gradient for a partially observed system, you could do so and you would get exactly the same equation.
51. Now, for a partially observed system, the trajectory distribution now would be a distribution over states, actions, and observations, and you have to marginalize out the states.
52. So the derivation for this is a little bit more involved, but you can do it at home.
53. However, if you follow through that derivation, you will end up with exactly the same equation that we got before, only the s's will be replaced by o's.
54. What this means is that you can use policy gradients in partially observed MDPs without any modification.
55. Just use them, and for this version of the policy gradient algorithm, it'll work just fine, insofar as regular policy gradients work.
56. Okay, now I mentioned before that maybe policy gradients, as I've described them so far, won't necessarily work very well.
57. If you actually try to implement them.
58. So what's wrong with a policy gradient?
59. Well, here's one problem that we could think about.
60. Let's say that the horizontal axis here denotes the trajectory, and I know the trajectory in general is not one-dimensional, but let's pretend it is, and the vertical axis represents the reward.
61. So here we have a reward, it's kind of this bell curve shape with a peak here, and let's say that we have three samples and the height of the bars here represents the reward of those samples.
62. So the blue curve shows the probability under the policy, that's the bell curve, and the green bars show the rewards.
63. So I apologize here, the Y-axis is actually a little bit overloaded, it's showing both rewards AND probabilities, so the blue thing is a probability, is a probability, it's always positive.
64. The green stuff is the reward, which may be positive or negative.
65. Okay, so with these three samples, we could now imagine when we calculate the policy gradient, which way will the blue policy distribution move?
66. Which way will the projection distribution move?
67. So take a moment to think about this.
68. Now, the policy gradient, you can think of it as basically a weighted maximum likelihood gradient.
69. So we're going to take each of these three points, and we're going to calculate log pi at each of these three points, and we'll multiply it by the value of the reward.
70. So the sample on the left has a very negative reward, so we will try to decrease the log probability there, and the two samples on the right have small but positive rewards, so we'll somewhat increase their probabilities.
71. So that means that the policy distribution will slide to the right, and it will mainly try to just look at the positive and the negative, and then we'll try to calculate the positive and the negative.
72. So let's really avoid that big negative sample.
73. Now, we know that if we take the reward function in MDP, and we offset it by a constant, meaning that we add the same constant to the rewards everywhere, the resulting optimal policy doesn't change, right?
74. This is for the same reason that if you have a maximization problem, let's say you're maximizing f of x, the maximum for f of x is the same even if you add a constant.
75. So the maximum for f of x is the same as the maximum for f of x.
76. So the maximum for f of x is the same even if you add a constant.
77. So the maximum for f of x is the same as the maximum for f of x.
78. plus 100, which is the same for f of x plus 1000.
79. So let's add a constant to the rewards.
80. So let's say that our rewards are now given by these bars.
81. Now, the relative rewards are exactly the same, so the samples on the right are still better than the samples on the left, but now I've added a constant to them, so they're all positive.
82. And now take a minute to imagine how the policy will change when we use these rewards.
83. With these rewards, we can now calculate the positive and the negative.
84. So we can calculate the positive and the negative.
85. So we can calculate the positive and the negative.
86. So we can calculate the positive and the negative.
87. So we can calculate the positive and the negative.
88. So with these rewards, of course, the policy will want of course, the policy will want of course, the policy will want to increase the log probabilities at all to increase the log probabilities at all to increase the log probabilities at all three samples, although it'll want to three samples, although it'll want to three samples, although it'll want to increase the ones on the right a bit increase the ones on the right a bit increase the ones on the right a bit more.
89. So maybe the policy will change more.
90. So maybe the policy will change more.
91. So maybe the policy will change like this.
92. Now you could imagine even like this.
93. Now you could imagine even like this.
94. Now you could imagine even more pathological changes to the reward.
95. more pathological changes to the reward.
96. What if I, for example, change the reward so that the two samples on the right actually go all the way to zero?
97. Or the sample on the left goes to zero?
98. This issue is actually an instance of high variance.
99. Essentially, the policy gradient estimator that we've described before has very high variance in terms of the samples that you get.
100. So depending on which samples you end up with, randomly, you might end up with very different values of the policy gradient for any finite sample size.
101. Now, as the number of samples goes to infinity, the policy gradient estimator will always yield the correct answer.
102. So this issue with adding constants to rewards will not make any difference.
103. But for finite sample sizes, they will.
104. And this makes policy gradients very hard to use.
105. It means that in practice, in order to make policy gradients be an effective tool for reinforcement learning, we must somehow lower this very high variance.
106. And a lot of advances in policy gradient algorithms basically revolve around different ways to reduce their variance.
107. And we'll cover some of those in today's lecture.
108. So you can think of an even more pathological version of this issue.
109. If some of the samples have a reward of zero, then their gradient basically doesn't matter at all.
110. And in general, this issue doesn't go away completely as you increase the number of samples, but it ends up being greatly mitigated.
111. All right.
112. So to review what we've covered so far, we talked about evaluating the RL objective with samples.
113. We talked about evaluating a policy gradient where we have to use this log gradient trick to remove the terms that we don't know, namely the initial state probability and the transition probability.
114. And then we can again evaluate the policy gradient using samples.
115. And we talked about how we can understand the policy gradient a little bit better intuitively by treating it as the formalization of trial and error learning and the evaluation of the policy gradient.
116. So we can then apply this algorithm to a gradient descent algorithm.
117. We briefly talked about how policy gradients can also handle partial observability.
118. And then lastly, we talked about why policy gradients might be hard to use.
119. So in the next portion of the lecture, we'll try to address this.