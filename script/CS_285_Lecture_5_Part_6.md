1. All right.
2. In the last portion of today's lecture, I'm going to discuss advanced policy gradients.
3. This material will go by a bit faster than the rest of the lecture, so if it's a little hard to follow, don't worry.
4. Please ask some questions in the comments, and we can discuss it more in class.
5. We will also have an entire other lecture later on in the course on even more advanced policy gradients materials.
6. So the particular issue that I want to discuss is a numerical issue that afflicts policy gradients, particularly in continuous action spaces.
7. To illustrate this issue, let me first describe an example problem.
8. Let's say that you have a one-dimensional state space.
9. So your state is essentially a number line, and your goal is to reach the state s equals zero.
10. You also have a one-dimensional action space.
11. So let's say that you're located at this state, and your actions can take place.
12. Either left or right.
13. Your reward is negative s squared minus a squared.
14. So you get a penalty based on squared distance from s zero, and you also get a penalty for taking large actions.
15. Your policy is going to be univariate and normally distributed with just two parameters.
16. One parameter k multiplies the state, so your mean is linear in the state, and the other parameter determines your variance, signal.
17. So yeah.
18. So you have k and sigma as your policy parameters θ.
19. So you can think of the policy as basically a little Gaussian centered at your current location, and your action is k times your current state.
20. So you're going to take this kind of noisy walk, hopefully towards the goal at s equals zero.
21. Now the convenient thing with having a two-dimensional parameter space is that we can actually visualize the entire vector field corresponding to the gradient at all locations in the parameter space.
22. This figure is borrowed from an excellent paper by Peters and Schall, which I'm going to cite at the end of this portion of the lecture.
23. The little blue arrow here shows the gradient, normalized to be unit length.
24. The horizontal axis is the first parameter k, and the vertical axis is the second parameter sigma.
25. The optimal setting for the parameters is k equals negative one and sigma equals zero.
26. So it's in the middle at the bottom of this plot.
27. But one of the things you might notice from looking at this plot is that the arrows don't actually point towards the optimum.
28. And the reason for this is that as sigma gets smaller and smaller, the gradient with respect to sigma gets larger and larger.
29. If you look at the form for the for the Gaussian probability, you'll notice this simply because the probability tracks as one over sigma squared.
30. So when you take the derivative, you get a one over sigma to the fourth term, which means that as sigma gets smaller the derivative gets larger.
31. The derivative with respect to k is still there, but the derivative with respect to sigma is larger.
32. So that when we renormalize the gradient, the sigma portion completely dominates as sigma gets smaller.
33. So that means that if we follow this gradient, it's going to take us a very very long time to reach the optimal parameter setting, because we'll spend all of our time just reducing sigma.
34. Now those of you that are familiar with numerical methods will probably recognize this as an issue of poor conditioning.
35. The intuition is that this is a essentially the same problem as optimizing, let's say, a quadratic function where the eigenvalues of the corresponding matrix have a very large ratio.
36. So if you have a quadratic function with some eigenvalues that are very large and some that are very small, then first-order gradient descent methods are really going to struggle in this kind of function.
37. This is essentially the same type of issue.
38. Now, again, if you have some background in numerical methods, at this point you might also be thinking, well, if the problem is poor conditioning, can we solve that problem by using a preconditioner?
39. And the answer is yes, and in fact what we're going to describe next could be viewed as a preconditioner.
40. But we're going to actually discuss it from a slightly different perspective, from the perspective of the dependence of your gradient on parameters.
41. So what I'm going to discuss next is how we can arrive at a covariant or natural policy gradient.
42. So here is the picture from the previous slide.
43. When we take a gradient step, via policy gradient, we take a gradient ascent step, choosing the step size for this type of gradient can be very delicate because some parameters affect the policy distribution a lot and some don't affect it very much.
44. So it's very hard to pick a single step size that works well both for k and for sigma, because the derivative with respect to sigma is going to get really really really large, whereas the one for k won't.
45. So what's really going on here is that different parameters affect the policy to different degrees.
46. Some parameters change the probabilities a lot, others don't change it very much.
47. But you want all of the parameters to reach their optimal value, so intuitively what you would like to do is to essentially have larger learning rates for those parameters that don't change the policy very much, and smaller learning rates for those that change it a lot.
48. If we want to view this a little bit more mathematically, one of the things we can do is look at the constraint optimization view of first-order gradient ascent.
49. So first-order gradient ascent, can be viewed as iteratively solving the following constraint optimization problem.
50. Take the argmax with respect to θ prime integrated with say a new parameter value.
51. Your objective is the first-order Taylor expansion of your original objective that's given by θ prime minus θ times grad j and you have a constraint that says θ prime minus θ squared should be small.
52. So it's like saying within an epsilon ball around your current parameter value find the parameter value that maximizes θ.
53. If you were to repeat what we just fabricate a sized y wszystkie parameter sauve at 1,0 if the parameter camera sogge, then you'll find we're at none.
54. So that's just a long way from a length price, of course that maximizes the linearization of your objective.
55. That's essentially what first-order gradient descent is doing, and you can think of alpha as basically the Lagrange multiplier for that constraint.
56. So those of you that have studied mirror descent or projected gradient descent would probably recognize this equation.
57. Usually we pick alpha rather than epsilon, but alpha is basically just the Lagrange multiplier that corresponds to epsilon.
58. So what this means is that when we do first-order gradient descent, we're finding the best value for θ prime within an epsilon ball, but that epsilon ball is in θ space.
59. Now our linearized objective is valid in only a small region around our current policy.
60. That's why we can't use very large step sizes.
61. But that region is very awkward to select if you have to select it in parameter space because some parameters will change the policy a lot and some will change it very little.
62. So intuitively what we would like to do is we would like to simplify the equation to the value of θ prime.
63. We would like to somehow reparameterize this process so that our steps are of equal size in policy space rather than parameter space, which would essentially rescale the gradient so that parameters that change the policy a lot get smaller rates, parameters that change the policy very little get larger rates.
64. So this is basically the problem.
65. This controls how far we go, and it's basically in the wrong space.
66. So can we rescale the gradient so that this doesn't happen?
67. What if we instead iteratively solve this problem, maximize the linearized objective, but subject to a constraint that the distributions don't change too much?
68. So here d is some measure of divergence between pi θ prime and pi θ, and we'd like that divergence measure to be less than or equal to epsilon.
69. So we'd like to pick some parameterization-independent divergence measure, a divergence measure that doesn't care about how you're parameterizing your policy, just which distribution it corresponds to.
70. A very good choice for this is the KL divergence.
71. The KL divergence is a standard divergence, and it's a very good choice for us.
72. So if I'm going to do a measurement of the divergence on distributions, I won't go into too much detail about how KL divergences are defined or derived, just that it's a measure of divergence on distributions, and it is parameter-independent, meaning that no matter how you parameterize your distributions, the KL divergence will remain the same.
73. Now, the KL divergence is a little complicated to plug into this kind of constrained optimization.
74. We want that constrained optimization to be very simple, because we're going to do that at every step of our gradient ascent procedure.
75. But if we take the second-order Taylor expansion, the KL divergence, around the point θ prime equals θ, then the KL divergence can be expressed as approximately as a quadratic form for some matrix F, right?
76. That's just what a second-order Taylor expansion is.
77. And it turns out that F is equal to what's called a Fisher information matrix.
78. The Fisher information matrix is the expected value under pi θ, that's your old policy, of ∇log π times ∇log π transpose.
79. So it's the expected value of the, the outer product of the gradient with itself.
80. Now notice that the Fisher information matrix is an expectation under pi θ, which should immediately suggest that we can approximate it by taking samples from pi θ, and actually trying to estimate this expectation.
81. And that's in fact exactly what we're going to do.
82. So now we've arrived at this formulation for our covariant policy gradient.
83. At every single step of our optimization, we maximize the linearity.
84. We've got this linearized objective, subject to this approximate divergence constraint, which is just the difference between θ prime and θ, under the matrix F.
85. So it's just like what we had before, θ prime minus θ, only before it was under the identity matrix, and now it's under the matrix F.
86. And if you actually write down Lagrangian for this, and solve for the optimal solution, you'll find that the solution is just to set the new θ to be θ.
87. So we have a solution.
88. We have a solution to be θ plus alpha, where alpha is the Lagrange multiplier, of F inverse times grad θ J(θ).
89. So before we had θ plus alpha grad θ J(θ).
90. Now we have θ plus alpha F inverse grad θ J(θ).
91. So F is basically our preposition right now.
92. And it turns out that if you apply this F inverse in front of your gradient, then your vector field changes in a very nice way.
93. So the picture on the right shows what you get by using this equation.
94. So the picture on the right shows what you get by using this equation.
95. So the picture on the right shows what you get by using this equation.
96. And now you can see that the red lines actually very nicely point towards the optimum.
97. And that means that you can converge a lot faster, and also you don't have to work nearly as hard at tuning your step size.
98. and also you don't have to work nearly as hard at tuning your step size.
99. Now there are a number of algorithms that use this trick.
100. The classical one, natural gradient or natural policy gradient, The classical one, natural gradient or natural policy gradient, selects alpha.
101. A more modern variant called trust region policy optimization selects epsilon and then derives alpha.
102. selects epsilon and then derives alpha.
103. So the way that you derive alpha is by solving for the optimal alpha at the same time while solving for f inverse grad θ J(θ).
104. We won't go into how to do this, but the high level idea is that by using conjugate gradient you can actually get both alpha and the natural gradient simultaneously.
105. So for more details on that, you can check out the paper called trust region policy optimization.
106. The takeaway from all of this is that the policy gradient can be numerically very difficult to use because different parameters are used.
107. Different parameters affect your distribution to very different degrees.
108. And you can address this by using the natural gradient, which simply requires multiplying your gradient by f inverse, where f inverse is an estimate of the Fisher information matrix.
109. And you can do this efficiently by using conjugate gradient.
110. Alright, a few notes on advanced policy gradient topics.
111. What more is there?
112. Well, next time we'll talk about actor-critic algorithms, where we'll introduce value functions and Q functions, and talk about how those can further decrease the variance of the policy gradient.
113. And then later in the class we'll talk more about natural gradient, automatic step size adjustment, and trust regions.
114. For now, let me briefly go over some papers that actually use policy gradients in interesting ways.
115. This is a paper actually by myself and Vladan Kolton from 2013, that used an off-policy version of policy gradient to incorporate examples.
116. So here, example demonstrations were incorporated using importance sampling, but unlike imitation learning, the policy wasn't just trying to copy the examples, it was actually trying to do better than those examples by using policy gradients.
117. And this used neural network policies for some locomotion tasks.
118. Here are some videos from the trust region policy optimization paper.
119. So this paper used a natural gradient with automatic step size adjustment, with both continuous and discrete actions.
120. And there was some code available for this if you want to check that out, from a paper from 2016 by Rocky Duan.
121. If you want to read more about policy gradients, here are some suggested readings.
122. The classical papers.
123. Reinforced was introduced in this paper by Williams in 1992.
124. This paper by Baxter and Bartlett introduced the, what I call the causality trick in the lecture.
125. They call it GPOMDP.
126. This is actually not the first paper to introduce it.
127. I'll actually mention the first paper when I talk about actual critic in the next lecture.
128. And this paper by Peters and Schall describes the natural gradient trick with some very nice illustrations.
129. Deep RL papers that use policy gradients.
130. The guided policy search paper that I mentioned before, which uses important sampled policy gradients.
131. This is the trust region policy optimization paper.
132. And then the PPO paper.
133. So these would be ones to check out if you're interested in policy gradients for Deep RL.