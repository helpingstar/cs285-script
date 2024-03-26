1. All right.
2. In the next section of today's lecture, I'm going to talk about another way that we can use the critic by incorporating the critic as a baseline to the policy gradient.
3. And this is going to have some interesting tradeoffs as compared to the standard Actor-Critic algorithm that we've discussed so far.
4. So on this slide, I have the equation for the Actor-Critic that we discussed in today's lecture, as well as the equation for the policy gradient that we saw in the previous lecture.
5. The Actor-Critic consists of the ∇log π term multiplied by the reward plus γ times the next value minus the current value.
6. The policy gradient consists of the ∇log π term times the sum of the rewards to go minus a baseline.
7. So the sum of the rewards to go is an unbiased single sample estimate of the Q value at s_{i,t} a_{i,t}.
8. Now, the Actor-Critic policy gradient estimator.
9. Has the advantage that drastically lowers the variance because the function approximator in the critic integrates in all those different possibilities for what might happen instead of relying on a single sample.
10. Unfortunately, the Actor-Critic gradient estimator also has the disadvantage that it's no longer unbiased.
11. The reason for that is that if your value function is slightly incorrect, which it might be because it's a function approximator trained on a finite number of samples.
12. Then you can't do that.
13. Then you can't do that.
14. You can't show anymore that in expectation, this gradient will actually converge to the true policy gradient.
15. The original policy gradient is unbiased.
16. So even though it might have high variance, even though each individual evaluation might be off, in expectation, it'll come out to the right value.
17. But it has much higher variance, which means that you typically need to use more samples or smaller learning rates.
18. So one question we could ask is, could we get a policy gradient that's unbiased?
19. Could we get a policy gradient estimator that is still unbiased, but uses the critic in some other way to lower the variance?
20. And the answer is that we can.
21. We can actually construct a policy gradient estimator that has slightly higher variance than the Actor-Critic version, but no bias like the policy gradient version.
22. And the way that we can do this is by using what's called a state-dependent baseline.
23. So it turns out that we can prove by extending the proof from the previous lecture that not only does the policy gradient, not only does the policy gradient, remain unbiased when you subtract any constant b, it actually still remains unbiased if you subtract any function that depends on only the state and not on the action.
24. It's actually a very easy proof to construct.
25. It follows almost exactly the proof that we had in the previous lecture.
26. And I would encourage those of you that are interested to go back and re-derive that to show that that is true.
27. It's very straightforward to do.
28. But the bottom line is that you can use any baseline that depends on the state, and a very good choice is the value function.
29. Because you would expect this single sample estimator in expectation to come out to be equal to the value function.
30. So if you use the value function as the baseline, then the numbers that are multiplying the ∇log π, should it in expectation be smaller, which means that their variance is smaller, which means that the variance of your entire policy gradient is smaller.
31. So it's actually quite a good idea to use a value function as a baseline.
32. Now this doesn't lower the variance as much as the full Actor-Critic algorithm, because you still have the sum over future rewards, but it's much lower than a constant baseline, and it's still unbiased.
33. Now, some of you might be wondering at this point, well, okay, we used the value function as a baseline, we made it depend on more stuff, and we got a lower variance, what if we make it depend on even more things?
34. What if we make the baseline depend on the state and the action?
35. Will we get an even lower variance that way?
36. And the answer is yes, but at that point things get much more complicated.
37. So what if we make the baseline depend on the state and the action?
38. Well, that's a little complicated.
39. So that's what we'll talk about next.
40. Methods that use state and action-dependent baselines are sometimes referred to as control variants in the literature.
41. So the true advantages, the Q value minus the value function, our approximate advantage that we use in policy gradients when we have a state-dependent baseline, is the sum of all future rewards minus the current value.
42. So this is nice because it has lower variance, but we can make the variance even lower if we subtract the Q value.
43. So this version has no bias, but it has higher variance than the actor critic because of the single sample estimate.
44. If we subtract the Q value, this has the nice property that it actually goes to 0 if your critic is correct.
45. So if your critic is correct and your future behavior is not too random, then you'd expect these quantities eventually to converge to 0.
46. Unfortunately, if you plug in this advantage estimator into your policy gradient, the policy gradient is no longer correct.
47. It won't actually give you the right gradient because there's an error term that you have to compensate for.
48. See, unlike the standard baseline, which integrates to 0 in expectation, an action-dependent baseline no longer integrates to 0.
49. It integrates to an error term, and you have to account for that error term.
50. So if you incorporate a baseline that depends on both the state and action and account for the error term, then you get this equation.
51. This equation is a valid estimator for the policy gradient even if your baseline doesn't depend on the action.
52. But in that case, the second term basically vanishes.
53. The second term is equal to 0.
54. But if your baseline depends on the action, the second term is no longer 0.
55. So the first term is just your policy gradient with your baseline.
56. The second term is what's left over.
57. It's the expected value...
58. It's the gradient of the expected value under the policy of your baseline.
59. Now, some of you might be looking at this and thinking, well, have we really bought ourselves anything?
60. Yes, the first term is going to be small because it's going to go to 0.
61. But the second term looks a lot like the original policy gradient.
62. So is this really any better?
63. Well, it turns out that this is actually a lot better in some cases.
64. For example, in many cases, the second term can actually be evaluated very, very accurately.
65. If you have discrete actions, you can sum over all possible actions.
66. If you have continuous actions, then you can sample a very large number of actions because evaluating the expectation over actions doesn't require sampling new states.
67. So it doesn't require actually interacting with the world, which means that you can generate many more samples from the same state, something that we could not do before when we had to actually make entire rollouts.
68. And furthermore, in many continuous action cases, if you make a careful choice of the class of distributions and the class of Q functions, this integral also has an analytic solution.
69. For example, the expected value of a quadratic function under a Gaussian distribution has an analytic solution.
70. So in many cases, the second term can be evaluated in such a way that its variance is 0 or very close to 0.
71. And the first term has low variance because ^{Q} minus Q π is typically going to be a small number.
72. So this kind of trick can be used to provide a very low variance policy gradient, especially if you can get a good Q function estimator.
73. If you want to read more about these kinds of control variants and how you can use a critic without incurring bias, provided the second term can be evaluated, then check out this paper by Shixiang Gu called QPro.
74. Okay, so so far we talked about ways that we can use critics and get policy gradient estimators that are unbiased.
75. But can we also use critics and get policy gradients that are still biased but only a little bit biased?
76. So next we're going to talk about eligibility traces and end-step returns.
77. First, let's take a look at the advantages of the Montecarlo advantage estimator in an actual critic algorithm.
78. I'm going to denote this as A hat C.
79. So A hat C is the current reward plus the next value minus the current value.
80. And this, as we saw, has much lower variance than the policy gradient, but higher bias if the value is wrong, and it's always at least a little bit wrong.
81. We can also take a look at the Montecarlo advantage estimator that we use in policy gradient algorithms, which I've written out here using the value function as the baseline, to keep things consistent so that both of them have a minus V term.
82. So here we have a sum of future rewards minus our baseline, which in this case is the value function.
83. This has no bias, but it has much higher variance because of the single sample estimate.
84. So at this point you might wonder, well, can we get something in between?
85. Could we, for example, sum over the rewards over the next five time steps instead of infinite time steps, and then put the value at the end of that?
86. And it turns out that we can, and we can use this to get a very nice way to control the bias-variance tradeoff.
87. There are a couple of factors that make this interesting.
88. First, when you're using a discount, typically your reward will decrease over time because you're discounting it.
89. So that means that the bias that you get from the value function is much less of a problem if you put the value function not at the next time step, but further into the future, where the rewards are smaller.
90. On the other hand, the variance that you get from the single sample estimator is also much more of a problem further into the future.
91. To understand why that might be, consider making single sample predictions.
92. Let's say that you ask me, make a single sample prediction which city you'll be in in five minutes.
93. Well, if you ask me that question, my single sample will be Berkeley, because I'll be in Berkeley in five minutes.
94. But if you ask me which city will I be in in 20 years, I might say, okay, you want a single sample, let's say San Francisco.
95. But I really don't know, it could be anything.
96. So there are many more possibilities far into the future than close to the present.
97. So usually, if you have these many possible trajectories emanating from the current state, they'll branch out a lot more further into the future and they'll be clustered much closer together in the present.
98. Which means that you have much higher variance far in the future and much lower variance closer to the present.
99. So that means that if you're going to use your single sample estimator for a chunk of this trajectory, you'd rather use it close to the present and then cut it off before the variance gets too big and then replace it with your value function, which has much lower variance.
100. And the value function will then account for all these different possibilities but potentially with some bias.
101. So the way that you can do this is by constructing what's called an n-step return estimator.
102. So in an n-step return estimator, you sum up rewards until some time step n and then you cut it off and replace it with the value function.
103. So here's the answer.
104. So this is the equation for the n-step return estimator.
105. The first part looks a lot like AMC.
106. So you sum up your rewards, but now you don't sum them up until infinity.
107. You sum them from t until t plus n.
108. You still subtract off your baseline, but then you have that remaining chunk, everything from n plus 1 until infinity, and that you replace with your value function.
109. So you evaluate your reward function at st plus n and you multiply it by γ to the n.
110. And that's an n-step return estimator.
111. And oftentimes, the greater than 1 works better than n equals 1.
112. So n equals 1 is kind of the limiting case of this.
113. Generally, the larger n is, the lower the bias, because the coefficient in front of the value function gets smaller and smaller, but the variance is higher because you're summing over more time steps with that single sample estimator, which is the first term.
114. So the first term contributes variance.
115. The last term contributes bias.
116. And the larger the n is, the smaller the third term, the larger the first term, which means the larger n is, the lower the bias, the higher the variance.
117. But very often, the sweet spot is not at n equals 1 nor at n equals infinity, and you may want to use an intermediate value.
118. Now, the last trick that I'm going to discuss is a way to actually generalize the n-step return estimator and actually construct a kind of hybrid estimator that uses many different n-step returns.
119. So do we have to choose just one n?
120. Do we have to make this hard slice at a particular point in time?
121. What if we want to actually construct all possible n-step return estimators and average them together?
122. So here's the equation for the n-step return estimator.
123. We can construct a kind of fused estimator, which we're going to call A-G-A-E for generalized advantage estimation, which consists of a weighted average of all possible n-step return estimators with a weight for different n.
124. And the way that you can choose your weight is by utilizing that insight that you'll have more bias if you use small n, more variance if you use larger n.
125. So you mostly prefer cutting earlier because then you have less variance, but you want to keep around some of those traces that go further in the future.
126. So a pretty decent choice that leads to an especially simple algorithm is to use an exponential falloff.
127. It's to set the weight, wn, to be lambda to some power n-1.
128. And if you think this is reminiscent of a discount factor, it's because it really is.
129. So if you use an exponential falloff, which means that wn is lambda to the n-1, then it turns out there's a very convenient way to evaluate this infinite sum over all possible n-step returns.
130. If wn is lambda to the n-1, then you can write out A-G-A-E as the current reward, plus γ times one minus lambda of the next value, plus lambda of the next reward and its next value, and so on.
131. So at every time step, you have one minus lambda times the value function at that time step, plus lambda of the G-A-E estimator from there on out.
132. And that gives you this weighted sum of all possible n-step return estimators.
133. This gives you a formula to calculate A-G-A-E recursively, but it turns out that we can collect the terms and get an even simpler form by expressing the G-A-E estimator as simply γ times lambda to the power of (t' - t) times a quantity that I'm going to call delta t prime.
134. And delta t prime is just the reward at t prime plus the value at t prime plus one minus the value at t prime.
135. It's kind of like that single step advantage estimator at time step t prime.
136. So this is pretty cool.
137. If you just construct a weighted sum of these one-step advantage estimators at every time step, weighted by γ times lambda to the (t' - t), you recover exactly the sum of all possible n-step returns.
138. So this is going to allow you to trade off bias variance just by choosing this lambda, which acts very similarly to a discount.
139. So the larger lambdas look further in the future, smaller lambdas use value functions closer to the present.
140. So it has a very similar effect as the discount, which also maybe sheds some light on the role that discounts play in policy gradients.
141. So this suggests that even if we didn't have lambda, the role of γ would be a kind of bias variance trade-off.
142. And that's in fact what's the case.
143. So the discount can also be interpreted itself as a kind of variance reduction because smaller discounts will result in your Monte Carlo single sample estimator putting lower weight on rewards far in the future, which are exactly the rewards that you'd expect to have high variance.
144. Just using a discount, of course, introduces more bias if you use a small discount as opposed to the correct higher discount value.
145. But using the lambda mitigates that by replacing it with a value function.