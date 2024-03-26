1. So in the next portion of today's lecture, we're going to talk about how we can modify the policy gradient calculation to reduce its variance, and in this way actually obtain a version of the policy gradient that can be used as a practical reinforcement learning algorithm.
2. The first trick that we'll start with is going to exploit a property that is always true in our universe, which is causality.
3. Causality says that the policy at time t' can't affect the reward at another time step t if t is less than t'.
4. This is another way of saying that what you do now is not going to change the reward that you got in the past.
5. Now, it's important to note here that this is not the same as the Markov property.
6. The Markov property says that the state in the future is independent of the state in the past given the present.
7. The Markov property is sometimes true, sometimes not true, depending on your particular temporal process.
8. Causality is always true.
9. Causality just says that rewards in the past are independent of decisions in the present.
10. So this is not really an assumption.
11. This is always true for any process where time flows forward.
12. The only way this would not be true is if you had time travel and you could take an action or travel back into the past and change your action.
13. But we're not allowed to do that.
14. All right, so I'm going to claim that the policy gradient that I've derived so far does not actually make use of this assumtions.
15. And that it can be modified to utilize this assumption and thereby reduce variance.
16. You can take a moment to think about where this assumption might be introduced.
17. The way that we're going to see this is we're going to rewrite the policy gradient equation.
18. I've not changed it in any way.
19. I've simply rewritten it.
20. And what I've done here is I used the distributive property to distribute the sum over rewards into the sum over ∇log π.
21. So you can think of this as taking that first set of parentheses over the sum of ∇log π's and taking the outer parentheses and wrapping it around the rewards.
22. So this gives me the sum over all of my samples from i=1 to N times the sum over time steps from 1 to T of ∇log π at that time step multiplied by another sum over another variable t' from 1 to T of the rewards.
23. So that means that at every time step, I multiply the grand log probability of the action at that time step t by the sum of rewards over all time steps in the past, present, and future.
24. Now at this point, you might start imagining how causality fits into this.
25. We're going to change the log probability of the action at every time step based on whether that action corresponded to larger rewards in the present and in the future, but also in the past.
26. And yet we know that the action at time step t can't affect the rewards in the past.
27. So that means that those other rewards will necessarily have to cancel out an expectation, meaning that if we generate enough samples, eventually we should see that all the rewards at time steps t' less than t will average out to a multiplier of 0, and they will not affect the log probability at this time step.
28. And in fact we can prove that this is true.
29. The proof is somewhat involved so I won't go through it here, but once we show that this is true, then we can simply change the summation of rewards and instead of summing from t'=1 to T, simply sum from t'=t to T.
30. Basically discard all the rewards in the past because we know that the current policy can't affect them.
31. Now we know they'll all cancel out an expectation, but for a finite sample size, they wouldn't actually cancel out.
32. So for a finite sample size removing all those rewards from the past will actually change your estimator but it will still be unbiased.
33. So this is the only change that we made.
34. Now having made that change we actually end up with an estimator that has lower variance.
35. The reason it has lower variance is very simple.
36. We've removed some of the terms from the sum which means that the total sum is a smaller number and expectations of smaller numbers have smaller variances.
37. Now one aside that I might mention here is that this quantity is sometimes referred to as the reward to go.
38. You can kind of guess why that is.
39. It's the rewards from now until the end of time which means that it refers to the rewards that you have yet to collect.
40. Basically all the rewards except for the ones in the past or the reward to go.
41. And we sometimes use the symbol ^{Q}_{i,t} to denote the reward to go.
42. Now take a moment to think back to the previous lecture where we also used the symbol Q.
43. The reward to go ^{Q} here actually refers to an estimate of the same quantity as the Q function that we saw in the previous lecture.
44. We will get much more into this in the next lecture when we talk about Actor-Critic algorithms but for now we'll just use a similar symbol with a hat on top to denote that it's a single sample estimate.
45. Alright now the causality trick that I described before you can always use it.
46. You'll use it in homework too.
47. It reduces your variance.
48. There's another slightly more involved trick that we can use that also turns out to be very important to make policy gradients practical and it's something called a baseline.
49. So let's think back to this cartoon that we had where we collect some trajectories and we evaluate the rewards and then we try to make the good ones more likely than the bad ones less likely.
50. That seemed like a very straightforward elegant way to formalize trial and error learning as a gradient ascent procedure.
51. But is this actually what policy gradients do?
52. Well intuitively policy gradients will do this if the rewards are centered, meaning that the good trajectories have positive rewards and the bad trajectories have negative rewards.
53. But this might not necessarily be true.
54. What if all of your rewards are positive?
55. Then the green checkmark will be increased, its probability will be increased, the yellow checkmark will be increased a little bit, and the red X will be also increased but a tiny bit.
56. So intuitively it kind of seems like what we want to do is we want to center our rewards so the things that are better than average get increased and the things that are worse than average get decreased.
57. For example maybe we want to subtract a quantity from our reward which is the average reward.
58. So instead of multiplying ∇log p by r(τ) we multiply by (r(τ)-b) where b is the average reward.
59. This would cause policy gradients to align with our intuition.
60. This would make policy gradients increase the probability of trajectories that are better than average and decrease the probabilities of trajectories that are worse than average.
61. And then this would be true regardless of what the reward function actually is even if the rewards are always positive.
62. That seems very intuitive but are we allowed to do that?
63. It seems like we just arbitrarily subtracted our constant from all of our rewards.
64. Is this even correct still?
65. Well it turns out that you can show that subtracting a constant b from your rewards in policy gradient will not actually change the gradient in expectation although it will change its variance.
66. Meaning that for any b doing this trick will keep your gradient estimator unbiased.
67. Here's how we can derive this.
68. So we're going to use the same convenient identity from before.
69. Which is that p(τ) times ∇log p(τ) is equal to ∇p(τ).
70. And now we're going to substitute this identity in the opposite direction.
71. So what we're going to do is we're going to analyze ∇log p(τ) times b.
72. So if I take the difference (r(τ) - b) and I distribute ∇log p into it then I get a ∇log p times r term which is my original policy gradient minus a ∇log p times b term which is the new term that I'm at.
73. So let's analyze just that term.
74. It's the expected value of ∇log p times b which means that it's the integral of p(τ) times ∇log p(τ) times b.
75. And now I'm going to substitute my identity back in.
76. So using the convenient identity in the blue box over there I know this is equal to the integral of ∇p(τ) times b.
77. Now by linearity of the gradient operator I can take both the gradient operator and b outside of the integral.
78. So this is equal to b times the gradient of the integral over τ of p(τ).
79. But p(τ) is a probability distribution and we know that probability distributions integrate to 1 which means that this is equal to b times the gradient with respect to θ of 1.
80. But the gradient with respect to θ of 1 is 0 because 1 doesn't depend on θ.
81. Therefore we know that this expected value comes out equal to 0 in expectation.
82. But for a finite number of samples it's not equal to 0.
83. So what this means is that subtracting b will keep our policy gradient unbiased but it will actually alter its variance.
84. So subtracting a baseline is unbiased in expectation.
85. The average reward which is what I'm using here turns out to not actually be the best baseline but it's actually pretty good.
86. And in many cases when we just need a quick and dirty baseline we'll use average reward.
87. However we can actually derive the optimal baseline.
88. The optimal baseline is not used very much in practical policy gradient algorithms but it's perhaps instructive to derive it just to understand some of the mathematical tools that go into studying variance.
89. So that's what we're going to do in the next portion.
90. In the next portion we'll go through a mathematical calculation where we will actually derive the expression for the optimal baseline to optimally minimize variance.
91. So to start with we're going to write down variance.
92. So if you have the variance of some random variable X it's equal to the E[x^2] - E[x]^2.
93. So we can use the same equation to write down the variance of our policy gradient.
94. So here's our policy gradient.
95. The variance of the policy gradient is equal to the expected value of the quantity inside the bracket squared minus the whole expected value squared.
96. Now the second term here is just the the policy gradient itself, right, because we know that (r(τ) - b) in expectation ends up not making a difference.
97. So basically the actual expected value of ∇log p times (r - b) is the same as the expected value of ∇log p times r.
98. So we can just forget about the second term, changing r is not going to change its value in expectation.
99. So it's really only the first term that we care about.
100. Alright, I'm going to change my notation a little bit just to declutter it.
101. So I'll just use g(τ) in place of ∇log p(τ).
102. So if you see g at the bottom that's just ∇log p, I just wanted to write a shorter value.
103. So I know that the second term in the variance doesn't depend on b, but the first term does.
104. So then in order to find the optimal b, I'm going to write down the derivative dVar/db and solve for the best b.
105. So the derivative of the second part is 0 because it doesn't depend on b.
106. So I just use the first part d/db of the expected value of g^2 times (r - b)^2.
107. Now I can expand out the quadratic form and I get d/db of the E[g^2 r^2] - 2 E[g^2 r b] + b^2 E[g^2].
108. So all I've done here is I've just expanded out the quadratic form (r - b)^2, distributed the g^2 into it, and then pulled constants out of expectations.
109. Now looking at this equation, we can see the first term doesn't depend on b, but the second two terms do.
110. So we can eliminate this part, and the second two terms if we take the derivative respect to b, the minus two term is linear in b and the plus term is quadratic in it, so we get the derivative is equal to -2E[g^2 r] + 2b × E[g^2].
111. Now we can push the constant term on the right-hand side and solve for b and we get this equation, b is equal to the E[g^2 r]/E[g^2].
112. So I've just solved for b when the derivative is equal to 0, so this is the optimal value of b.
113. Now looking at this thing you could try to imagine what is the optimal baseline really intuitively.
114. Well perhaps one thing that might jump out at you is that the baseline now actually depends on the gradient, which means that if the gradient is a vector with multiple dimensions, if you have multiple parameters, you'll actually have a different baseline for every entry in the gradient.
115. So if you have a hundred different policy parameters you'll have one value of the baseline for parameter one, a different value of the baseline for parameter two and intuitively looking at this equation the baseline for each parameter value is basically the expected value of the reward weighted by the magnitude of the gradient for that parameter value.
116. So it's a kind of reweighted version of the expected reward, it's not the average reward anymore, it's a reweighted version of it.
117. It's reweighted by gradient magnitudes.
118. So this is the baseline that minimizes the variance.
119. Now again, in practice, we often don't use the optimal variance, we just, sorry, we often don't use the optimal baseline.
120. We typically just use the expected reward.
121. But if you wanted the optimal baseline, this is how you would get it.
122. All right, so to review what we've covered so far, we talked about the high variance of policy gradients algorithms.
123. We talked about how we can lower that variance by exploiting the fact that present actions don't affect past rewards.
124. And we talked about how we can use baselines, which are also unbiased, and we can analyze variance to solve for the optimal baseline.