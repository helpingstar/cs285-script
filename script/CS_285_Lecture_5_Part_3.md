1. So in the next portion of today's lecture, we're going to talk about how we can modify the policy gradient calculation to reduce its variance, and in this way actually obtain a version of the policy gradient that can be used as a practical reinforcement learning algorithm.
2. The first trick that we'll start with is going to exploit a property that is always true in our universe, which is causality.
3. Causality says that the policy at time t' can't affect the reward at another time step t if t is less than t'.
4. This is another way of saying that what you do now is not going to change the reward that you got in the past.
5. Now, it's important to note here that this is not the same as the Markov property.
6. The Markov property says that the state in the future is independent of the state in the past given the present.
7. The Markov property is sometimes true, sometimes not.
8. Sometimes not true, depending on your particular temporal process.
9. Causality is always true.
10. Causality just says that rewards in the past are independent of decisions in the present.
11. So this is not really an assumption.
12. This is always true for any process where time flows forward.
13. The only way this would not be true is if you had time travel and you could take an action or travel back into the past and change your action.
14. But we're not allowed to do that.
15. All right, so I'm going to claim that the policy gradient that I've derived so far does not actually make use of this.
16. And that it can be modified to utilize this assumption and thereby reduce variance.
17. You can take a moment to think about where this assumption might be introduced.
18. The way that we're going to see this is we're going to rewrite the policy gradient equation.
19. I've not changed it in any way.
20. I've simply rewritten it.
21. And what I've done here is I used the distributive property to distribute the sum over rewards into the sum over gratitudes.
22. And I've also used the distributive property to distribute the sum over gratitudes.
23. So this is the sum over all of my samples from i equals 1 to n.
24. So you can think of this as taking that first set of parentheses over the sum of grad log pi's and taking the outer parentheses and wrapping it around the rewards.
25. So this gives me the sum over all of my samples from i equals 1 to n times the sum over time steps from 1 to capital T of grad log pi at that time step multiplied by another sum over another variable t prime from 1 to capital T of the rewards.
26. So that means that at every time step, I multiply the grand log probability of the action at that time step t by the sum of rewards over all time steps in the past, present, and future.
27. Now at this point, you might start imagining how causality fits into this.
28. We're going to change the log probability of the action at every time step based on whether that action corresponded to larger rewards in the present and in the future, but also in the past.
29. And yet we know that the action at time step t can't affect the rewards in the past.
30. So that means that those other rewards will necessarily have to cancel out an expectation, meaning that if we generate enough samples, eventually we should see that all the rewards at time steps t prime less than t will average out to a multiplier of 0, and they will not affect the log probability at this time step.
31. And in fact we can prove that this is true.
32. The proof is somewhat involved so I won't go through it here, but once we show that this is true, then we can simply change the summation of rewards and instead of summing from t prime equals 1 to capital T, simply sum from t prime equals t to capital T.
33. Basically discard all the rewards in the past because we know that the current policy can't affect them.
34. Now we know that they'll all cancel out an expectation, but for a finite sample size, they wouldn't actually cancel out.
35. So for a finite sample size removing all those rewards from the past will actually change your estimator but it will still be unbiased.
36. So this is the only change that we made.
37. Now having made that change we actually end up with an estimator that has lower variance.
38. The reason it has lower variance is very simple.
39. We've removed some of the terms from the sum which means that the total sum is a smaller number and expectations of smaller numbers have smaller variances.
40. Now one aside that I might mention here is that this quantity is sometimes referred to as the reward to go.
41. You can kind of guess why that is.
42. It's the rewards from now until the end of time which means that it refers to the rewards that you have yet to collect.
43. Basically all the rewards except for the ones in the past or the reward to go.
44. And we sometimes use the symbol Q hat i comma t to denote the reward to go.
45. Now take a moment to think back to the previous lecture where we also used the symbol Q.
46. The reward to go Q hat here actually refers to an estimate of the same quantity as the Q function that we saw in the previous lecture.
47. We will get much more into this in the next lecture when we talk about actual critic algorithms but for now we'll just use a similar symbol with a hat on top to denote that it's a single sample estimate.
48. Alright now the causality trick that I described before you can always use it.
49. You'll use it in homework too.
50. It reduces your variance.
51. There's another slightly more involved trick that we can use that also turns out to be very important to make policy gradients practical and it's something called a baseline.
52. So let's think back to this cartoon that we had where we collect some trajectories and we evaluate the rewards and then we try to make the good ones more likely than the bad ones less likely.
53. That seemed like a very straightforward elegant way to formalize trial and error learning as a gradient-assent procedure.
54. But is this actually what policy gradients do?
55. Well intuitively policy gradients will do this if the rewards are centered, meaning that the good trajectories have positive rewards and the bad trajectories have negative rewards.
56. But this might not necessarily be true.
57. What if all of your rewards are positive?
58. Then the green checkmark will be increased, its probability will be increased, the yellow checkmark will be increased a little bit, and the red X will be also increased but a tiny bit.
59. So intuitively it kind of seems like what we want to do is we want to center our rewards so the things that are better than average get increased and the things that are worse than average get decreased.
60. For example maybe we want to subtract a quantity from our reward which is the average reward.
61. So instead of multiplying grad log P by R of tau we multiply by R of tau minus R of tau.
62. So we get R minus B where B is the average reward.
63. This would cause policy gradients to align with our intuition.
64. This would make policy gradients increase the probability of trajectories that are better than average and decrease the probabilities of trajectories that are worse than average.
65. And then this would be true regardless of what the reward function actually is even if the rewards are always positive.
66. That seems very intuitive but are we allowed to do that?
67. It seems like we just arbitrarily subtracted our constant from all of our rewards.
68. Is this even correct still?
69. Well it turns out that you can show that subtracting a constant B from your rewards in policy gradient will not actually change the gradient in expectation although it will change its variance.
70. Meaning that for any B doing this trick will keep your gradient estimator unbiased.
71. Here's how we can derive this.
72. So we're going to use the same convenient identity from before.
73. So we're going to use the same thing in this case.
74. Which is that P of tau times grad log P of tau is equal to grad P of tau.
75. And now we're going to substitute this identity in the opposite direction.
76. So what we're going to do is we're going to analyze grad log P of tau times B.
77. So if I take the difference R of tau minus B and I distribute grad log P into it then I get a grad log P times R term which is my original policy gradient minus a grad log P times B term which is the new term that I'm at.
78. So let's analyze just that term.
79. It's the expected value of grad log P times B which means that it's the integral of P of tau times grad log P of tau times B.
80. And now I'm going to substitute my identity back in.
81. So using the convenient identity in the blue box over there I know this is equal to the integral of grad P of tau times B.
82. Now by linearity of the gradient operator I can take both the gradient operator and B outside of the integral.
83. So this is equal to the gradient operator and B outside of the integral.
84. So this is equal to B times the gradient of the integral over tau of P of tau.
85. But P of tau is a probability distribution and we know that probability distributions integrate to 1 which means that this is equal to B times the gradient with respect to theta of 1.
86. But the gradient with respect to theta of 1 is 0 because 1 doesn't depend on theta.
87. Therefore we know that this expected value comes out equal to 0 in expectation.
88. But for a finite number of samples it's not equal to 0.
89. So what this means is that subtracting B will keep our policy gradient unbiased but it will actually alter its variance.
90. So subtracting a baseline is unbiased in expectation.
91. The average reward which is what I'm using here turns out to not actually be the best baseline but it's actually pretty good.
92. And in many cases when we just need a quick and dirty baseline we'll use average reward.
93. However we can actually derive the optimal baseline.
94. The optimal baseline is not used very much in practical policy gradient algorithms but it's perhaps instructive to derive it just to understand some of the mathematical tools that go into studying variance.
95. So that's what we're going to do in the next portion.
96. In the next portion we'll go through a mathematical calculation where we will actually derive the expression for the optimal baseline to optimally minimize variance.
97. So to start with we're going to write down variance.
98. So if you have the variance of some random variable X it's equal to the expected value of 0, it's equal to the optimal variance for each of x squared minus the expected value of x squared.
99. So we can use the same equation to write down the variance of our policy gradient.
100. So here's our policy gradient.
101. The variance of the policy gradient is equal to the expected value of the quantity inside the bracket squared minus the whole expected value squared.
102. Now the second term here is just the the policy gradient itself, right, because we know that r of tau minus b in expectation ends up not making a difference.
103. So basically the actual expected value of grad log p times r minus b is the same as the expected value of grad log p times r.
104. So we can just forget about the second term, changing r is not going to change the value of grad log p times r.
105. Now the fact that the expectation value of grad log p its value in expectation.
106. So it's really only the first term that we care about.
107. Alright, I'm going to change my notation a little bit just to declutter it.
108. So I'll just use g of tau in place of grad log p of tau.
109. So if you see g at the bottom that's just grad log p, I just wanted to write a shorter value.
110. So I know that the second term in the variance doesn't depend on b, but the first term does.
111. So then in order to find the optimal b, I'm going to write down the derivative dvar db and solve for the best b.
112. So the derivative of the second part is 0 because it doesn't depend on b.
113. So I just use the first part ddb of the expected value of g squared times r minus b squared.
114. Now I can expand out the quadratic form and I get ddb of the expected value of g squared r squared minus b squared.
115. So I can write down the derivative of the second part of the equation and I just have to find the optimal b value of g of root 2 times the expected value of g squared rb plus b squared times the expected value of g squared.
116. So all I've done here is I've just expanded out the quadratic form r minus b squared, distributed the g squared into it, and then pulled constants out of expectations.
117. Now looking at this equation, we can see the first term doesn't depend on b, but the second two terms do.
118. So we can eliminate this part, and the second two terms if we take the equation from the first terms we do see that the first two terms are good.
119. So that's the first 나는 we can also eliminate this part.
120. And the second two terms, if we take their derivative with respect to b, the term is linear in b and the plus term is quadratic in it, so we get the derivative is equal to negative 2 times the expected value of g squared r plus 2b times the expected value of g squared.
121. Now we can push the constant term on the right-hand side and solve for b and we get this equation, b is equal to the expected value of g squared r divided by the expected value of g squared.
122. So I've just solved for b when the derivative is equal to 0, so this is the optimal value of b.
123. Now looking at this thing you could try to imagine what is the optimal baseline really intuitively.
124. Well perhaps one thing that might jump out at you is that the baseline now actually depends on the gradient, which means that if the gradient is a vector with multiple dimensions, if you have multiple parameters, you'll actually have a different baseline for every entry in the gradient.
125. So if you have a hundred different policy parameters you'll have one value of the baseline for parameter one, a different value of the baseline for parameter two and intuitively looking at this equation the baseline for each parameter value is basically the expected value of the reward weighted by the magnitude of the gradient for that parameter value.
126. So it's a kind of reweighted version of the expected reward, it's not the average reward anymore, and that's it.
127. it's a reweighted version of it.
128. It's reweighted by gradient magnitudes.
129. So this is the baseline that minimizes the variance.
130. Now again, in practice, we often don't use the optimal variance, we just, sorry, we often don't use the optimal baseline.
131. We typically just use the expected reward.
132. But if you wanted the optimal baseline, this is how you would get it.
133. All right, so to review what we've covered so far, we talked about the high variance of policy gradients algorithms.
134. We talked about how we can lower that variance by exploiting the fact that present actions don't affect past rewards.
135. And we talked about how we can use baselines, which are also unbiased, and we can analyze variance to solve for the optimal baseline.