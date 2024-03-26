1. In today's lecture, we're going to cover Actor-Critic algorithms.
2. Actor-critic algorithms build on the policy gradient framework that we discussed in the previous lecture, but they also augment it with learned value functions and Q functions.
3. So to begin, let's recap the policy gradients material from last time.
4. Last time we learned about the REINFORCE algorithm, which alternates between three steps.
5. It samples a batch of trajectories by running the current policy in the environment.
6. So you sample n trajectories in this way.
7. Then we use these trajectories to compute an estimate of the policy gradient, which is calculated by averaging over all of our samples a sum over all time steps of ∇log π at that time step times the sum of rewards from that step until the end, or the reward to go.
8. And then we take this approximate policy gradient and we add it multiplied by some learning rate to our current parameter vector, which corresponds to a gradient ascent optimization process.
9. This algorithm follows the basic anatomy of the reinforcement learning algorithm that we discussed before, where the orange box corresponds to generating the samples, the green box corresponds to calculating the reward to go at every time step for every sample, and the blue box corresponds to applying the gradient ascent rule.
10. Now in the lecture last time, I somewhat suggestively used the symbol ^{Q} to denote the reward to go.
11. And this choice was deliberate, because when you exploit this causality property that I described, it turns out that the way that you should calculate your policy gradient is by multiplying each ∇log π by the total reward that you expect to get if you start in state s_{i,t}, then take action a_{i,t}, and then follow your policy.
12. That's a very reasonable interpretation of the policy gradient.
13. You're essentially saying that you will increase the probability of those actions that in expectation lead to higher rewards and decrease the probabilities of those actions that in expectation lead to lower rewards.
14. But let's examine this ^{Q} term a little bit more closely.
15. ^{Q} represents an estimate of the expected reward if you take action a_{i,t} in state s_{i,t} and then follow your policy until the end of the trajectory.
16. But can we get a better estimate of this quantity?
17. Let's imagine that this curvy line represents one of the trajectories that you sampled, and your ^{Q} is calculated at a particular time step.
18. So this green circle represents the state s_{i,t}, so the i-th sample times step t, and at that point in time we're going to calculate an estimate of our reward-to-go ^{Q}.
19. And then we're going to multiply our ∇log π by that reward-to-go.
20. So the way that we calculate this estimate is by summing up the rewards that we actually got along that trajectory.
21. But that trajectory represents just one of the many possibilities, so if we were to somehow accidentally land in the same exact state again and then run our policy just like we did on this rollout, we might get a different outcome, simply because the policy and the MDP have some randomness in them.
22. So right now we're using a single step estimate for the reward to go, but in reality there are many possibilities for what might happen next.
23. So we would have a better estimate of the reward to go if we could actually compute a full expectation over all these different possibilities.
24. The reason that there are many possibilities is simply because there's randomness in the system.
25. Our policy has randomness and our MDP has randomness.
26. But this randomness can be quite significant, which means that our single sample estimate that we got by summing up the rewards that we actually obtained in that trajectory might be quite far off from the actual expected value.
27. Now this problem I'm going to claim directly relates to the high variance of the policy gradient.
28. And I'd like all of you to take a moment to think about what this has to do with variance.
29. So the connection to variance is that the policy gradient way of calculating the reward to go is a single sample estimate of a very complex expectation.
30. The fewer samples you use to estimate an expectation, the higher the variance of your estimator will be.
31. So a single sample estimator has very high variance.
32. If we could somehow generate a million samples starting from the same state action tuple, then we would have much lower variance.
33. If we could somehow calculate this expectation exactly, we would have much, much lower variance.
34. So if we had access to the true expected reward to go, defined as the true expected value of the sum of rewards that we get starting from state s_{i,t} and action a_{i,t}, then the variance of our policy gradient would be much lower.
35. And then if we had this Q function, we could simply plug it in in place of ^{Q} and get a lower variance policy gradient.
36. Now in the previous lecture, we also learned about this thing called baselines, which could lower the variance of a policy gradient even further.
37. Can we apply a baseline even when we have the true Q function?
38. And the answer is that of course we can.
39. So we can subtract some quantity b.
40. We learned last time that the average reward was a good choice for b, although not the optimal choice.
41. So what do we average?
42. Well, we could average Q values.
43. So we could say, well let's make b just be the average Q value at that time step over all the states and actions that we saw, and then we will have this appealing property that the policy gradient will increase the probability of actions that are better than average in terms of their reward to go expectation and decrease the probability of actions that are worse than average.
44. But it turns out that we can lower the variance even further because the baseline can actually depend on the state.
45. It can't depend on the action that leads to bias, but you can make it depend on the state.
46. So, if you make the baseline depend on the state, then the best thing to do, or not the optimal thing to do, but a better thing to do, would be to compute the average reward over all the possibilities that start in that state.
47. So not just the average reward over all possibilities of that time step, but specifically in that specific state.
48. And if you average your Q values over all the actions in a particular state, that's simply the definition of the value function.
49. So a very good choice for the baseline is the value function.
50. So you can calculate your policy gradient as ∇log π multiplied by Q(s_{i,t},a_{i,t}) minus V(s_{i,t}).
51. This is in fact a very intuitive quantity because the difference between the Q value and the value function represents your estimate of how much better the action a_{i,t} is on average than the average action you would take in the state s_{i,t}.
52. So it makes a lot of sense to multiply your ∇log π terms by this because it's directly saying take the actions that are better than average in that state and increase their probability and take the actions that are worse than average in that state and decrease their probability.
53. In fact, this (Q - V) term is so important that we have a special name for it.
54. We call it the advantage function.
55. The reason we call it the advantage function is that it represents how advantageous the action a_{i,t} is as compared to the average performance that you would expect the policy π_θ to get in the state s_{i,t}.
56. Okay.
57. So let's talk about state and state action value functions.
58. By the way, when I say state action value function or Q function, those mean exactly the same thing.
59. But sometimes saying state action value function can be a little clearer.
60. So our Q function or state action value function represents the total expected reward that you expect to get if you start in state s_t, take action a_t, and then follow your policy.
61. We will often write the Q function with the superscript π to emphasize that the Q function depends on π.
62. So every policy will have a different Q function.
63. The value function is the expected value over all the actions in state s_t under your current policy of the Q value.
64. Another way of saying it is that it's the total reward that you expect to get if you start in state s_t and then follow your policy.
65. The advantage function is the difference between these two quantities.
66. And the advantage function represents how much better the action a_t is as compared to the average performance of your policy π in state s_t.
67. So we can get a very good estimate of the policy gradient if we simply multiply the ∇log π terms by the advantage value at s_{i,t} a_{i,t}.
68. Now of course in reality we won't have the correct value of the advantage.
69. We'll have to estimate it, for example using some function approximator.
70. So the better our estimate of the advantage, the lower our variance will be.
71. Now it's also worth mentioning that the kind of Actor-Critic methods that we'll discuss in today's lecture don't necessarily produce unbiased estimates of the advantage function.
72. So while the policy gradient we've discussed so far has been unbiased, if your advantage function is incorrect, then your entire policy gradient can also be biased.
73. Usually we're okay with that because the enormous reduction in variance is often worth the slight increase in bias that we incur from using approximate Q values and value functions.
74. So to summarize, the conventional policy gradient uses a kind of a Monte Carlo estimate of the advantage calculated by using the one sample that you have in the remainder of the current trajectory by summing up the rewards in your trajectory and subtracting a baseline.
75. This is an unbiased but high-variance single sample estimate, and we can replace it with an approximate advantage function, which itself is usually calculated from an approximate Q function or an approximate value function, and get a much lower variance estimate, because now we're potentially getting a better estimate for this expectation that does not rely on a single sample, but often the resulting approximate value functions will have some bias, so we'll trade off lots of variance for some small increase in bias.
76. So the structure of the resulting algorithms will now have a much more elaborate green box.
77. So the orange box will be the same as before, we'll generate samples by running our policy.
78. The blue box will still be the same, we'll still use the policy gradient to do gradient descent, but the green box will now involve fitting some kind of estimator, either an estimator to Q^π, or an estimator to V^π, or A^π.
79. So let's talk about that next.
80. Let's talk about fitting value functions.
81. So we have three possible quantities, Q, V, or A.
82. Ultimately we want A, but the question we might ask is, well what should we fit?
83. Which of these three should we fit?
84. And what should we fit to?
85. What should our targets be?
86. So should we fit Q, V, or A?
87. Take a moment to think about this choice, and consider some of the pros and cons of one choice or another.
88. So the Q function is the expected value of the reward that we will get when we start from state s_t, take action a_t, and then follow our policy.
89. Now one very convenient property of this is that because s_t and a_t are not actually random variables, we can rewrite the Q function as simply the current reward plus the expected value of the reward in the future.
90. Because the current reward depends on s_t and a_t, and they are not random.
91. So this equality is exact.
92. And this quantity that we're adding is simply the expected value of the value function at the state s_{t+1} that we will get when we take action a_t in state s_t.
93. So we can similarly write the Q function in terms of the value function as the current reward plus the expected value of the reward of the value function of the next time step.
94. And the expectation here is of course taken with respect to the transition dynamics.
95. Now we can make a small approximation where we could say that the actual state s_{t+1} that we saw in the current trajectory is kind of representative of the average s_{t+1} that we will get.
96. Now at this point we've made an approximation.
97. This is not an exact equality.
98. We're essentially approximating the distribution over states at the next time step with, again, a single sample estimator.
99. But now it's a single sample estimator for just that one time step.
100. Everything after that is still integrated out as represented by the value function V^π.
101. So we've made this approximation, and now we might wonder, well, okay, so we lost a little bit.
102. We still have lower variance, but not quite as low as we had before.
103. Why would we want to do that?
104. Well, the reason that we would want to do that is because if we then substitute this approximate equation for the Q value into the equation for the advantage, we get this very appealing expression where the advantage is now approximately equal to the current reward plus the next value minus the current value.
105. This is still an approximation because, to be exact, this V^π(s_{t+1}) needs to be in expectation over all possible values of s_{t+1}, whereas we've just substituted the actual s_{t+1} that we saw.
106. But what's very appealing about this equation is that now it depends entirely on V.
107. And V is more convenient to learn than Q or A, because Q and A both depend on the state and the action, whereas V depends only on the state.
108. When your function approximator depends on fewer things, it's easier to learn because you won't need as many samples.
109. So maybe what we should do is just fit V^π(s).
110. This is not the only choice for Actor-Critic algorithms, and we will learn about Actor-Critic methods that use Q functions as well later on in the course, but for now, we'll talk about Actor-Critic algorithms that just fit V^π(s), and then use the equation to derive the advantage function approximately.
111. So when we fit V^π(s), we would have some kind of model, such as a neural network, that maps states s to approximate values ^{V}^π(s).
112. And this network will have some parameters, which I'm going to call ϕ.
113. So let's talk about the process of fitting V^π(s).
114. This process is sometimes referred to as policy evaluation, because V^π represents the value of the policy at every state, so calculating the value is evaluation.
115. In fact, if you think back to the lecture last week on the definitions of reinforcement learning problems, you will remember that the reinforcement learning objective itself can be expressed as the expected value of the value function over the initial state distribution.
116. So if you compute the value function, you can literally evaluate how good your policy is just by averaging together the values at the initial states.
117. So that's the expression here.
118. Our objective J(θ) can be expressed as just the expected value of V^π at the initial states.
119. So how can we perform policy evaluation?
120. Well, one thing that we could do is we could use Monte Carlo policy evaluation.
121. In a sense, this is what policy gradients do.
122. In Monte Carlo policy evaluation, we earn our policy many, many times, and then sum together the rewards obtained along the trajectories generated by the policy, and use that as an unbiased but high-variance estimate of the policy's total reward.
123. So we could say that the value at state s_t is approximately the sum over all the rewards that we saw after visiting state t along the trajectory that visited state s_t.
124. So here is our rollout, here is the state s_t, and we're just going to sum all of the things that we saw after that along that one trajectory.
125. Now ideally, what we would like to be able to do is sum over all possible trajectories that could occur when you start from that state, because there's more than one possibility.
126. So we would like to sum over all of these things.
127. Unfortunately, in the model-free setting, this is generally impossible, because this requires us to be able to reset back to the state s_t, and run multiple trials starting from that state.
128. Generally, we don't assume that we're able to do this.
129. We only assume that we're able to run multiple trials from the initial state.
130. So typically we can't do this, but if you have access to a simulator that you can reset, you can technically calculate your Monte Carlo values in this way.
131. Okay.
132. So what happens if we use a neural network function approximator for the value function with this kind of Monte Carlo evaluation scheme?
133. Well, we have our neural network, ^{V}^π with parameters ϕ.
134. We're going to, at every state that we visit, sum together the remaining rewards, and that will produce our target values.
135. But then, instead of plugging those reward-to-goes directly into our policy gradient, we'll actually fit a neural network to those values.
136. And that will actually reduce our variance, because even though we can't visit the same state twice, our function approximator, our neural network, will actually realize that different states that we visit in different trajectories are similar to one another.
137. So even though this green state along the first trajectory will never be visited more than once in continuous state spaces, if we have another trajectory rollout that is kind of nearby, but then where something else happened later down the line in that trajectory, the function approximator will realize that these two states are similar, and when it tries to estimate the value at both of these states, the value of one will sort of leak into the value of the other.
138. That's essentially generalization.
139. Generalization means that your function approximator understands that nearby states should take on similar values.
140. So if you accidentally had a very different outcome in one of those states than you did in the other, the function approximator will, to some degree, average those out, and produce lower variance estimates than you would have gotten if you're just directly used that single sample value in your policy gradient.
141. So it's not good as making multiple rods from same state, but it's still pretty good.
142. So the way that we would do this is we would generate training data by taking all of our rollouts, and for each state along every rollout we create a tuple consisting of the state s_{i,t} and a label corresponding to the sum of rewards that we saw starting from s_{i,t}, for the rest of that rollout.
143. And we're going to call these labels y_{i,t}.
144. And when I say target value, I mean y_{i,t}.
145. So we'll get these tuples s_{i,t}, y_{i,t}, and then we'll solve a supervised regression problem.
146. We'll train our neural network value function so that its parameters ϕ minimize the sum over all of our samples of the squared error between the value function's prediction, and the single sample Monte Carlo estimate of the value at that state.
147. Of course, if our function approximator massively overfits and produces exactly the training label at every single state, then we wouldn't have gained much as compared to just directly using the y_{i,t} values in our policy gradient.
148. But if we get generalization, meaning that our function approximator understands that two nearby states should have similar values, even if their labels are different, then we'll actually get lower variance, because this function approximator will now average out the dissimilar labels at similar states.
149. But can we do even better?
150. So the ideal target that we would like to have when training our value function is the true expected value of rewards starting from the state s_{i,t}.
151. Of course, we don't know this quantity.
152. The Monte Carlo target that we used before uses a single sample estimate of this quantity.
153. But we can also use the relationship that we wrote out before, where we saw that a Q function is simply equal to the reward at the current time step plus the expected reward starting from the next time step.
154. And if we write out this quantity, then we can perform the same substitution as before and actually replace the second term in the summation with our estimate of the value function.
155. And this is a better lower variance estimate of the reward to go than our single sample estimator.
156. So this says, let's use the actual reward that we saw at the current time step plus the value at the actual state that we saw at the next time step.
157. Now, of course, we don't know the value V^π, so we're going to approximate that simply by using our previous function approximator.
158. So we'll assume that our previous ^{V}^π_ϕ was kind of okay, like maybe it wasn't great, but it's probably better than nothing.
159. So we can plug it in in place of V^π and get what is called a bootstrap estimator.
160. So here we're directly going to use the previous fitted value function to estimate this quantity.
161. So now our training data will consist of tuples of the states that we saw s_{i,t} and labels that correspond to the reward that we actually got at that time step r(s_{i,t},a_{i,t}) plus estimate of the value function at the actual next state, s_{i,t+1} that we saw.
162. Now this estimate ^{V} might be incorrect, but as we repeat this process, hopefully these values will get closer and closer to the correct values, and because the ^{V}s are averaging together all possible future returns, we expect their variance to be lower.
163. So now our target value y_{i,t} is given by the sum.
164. And our training process, just like before, is going to be supervised regression onto these y_{i,t}s.
165. This is sometimes referred to as a bootstrap estimate, and the bootstrap estimate has lower variance because it's using ^{V} instead of a single sample estimator, but it also has higher bias because our ^{V}^π_ϕ might be incorrect.
166. So that's the trade-off.
167. Alright, so to conclude this portion of the lecture, what I want to do is give you a few examples of policy evaluation just so that you get a better intuitive understanding of what the heck policy evaluation actually means.
168. Because in many cases, policy evaluation actually is a very intuitive concept.
169. For example, if we're training a reinforcement learning system with Actor-Critic to play backgammon, this is from the TD Gammon paper in 1992, maybe our reward corresponds to the outcome of the game.
170. It's 1 if you win the game, and 0 if you don't.
171. Then our value function is simply the expected outcome given the board state.
172. If you get a 1 if you win the game and 0 if you lose, then the value function just directly predicts the probability that you'll win the game given the state of the board right now.
173. Very intuitive.
174. Similarly, if you're training a system to play Go, and your reward is the game outcome, exactly the same thing.
175. Your value function is actually trying to predict how likely are you to win the game given the state of the board right now.
176. Now this is very convenient for board games because we know the rules that govern these board games, so we can simulate what would happen if we were to play a move.
177. In fact, we can simulate every possible move, check its value, and then take the move that leads to the highest value state.
178. This is much cheaper than doing a full game tree because all you have to do is predict one step into the future, and then your value function tells you given what will happen after that one step, how likely are you to win the game.
179. And then you take the move that most increases your probability to win the game.
180. So policy evaluation can have very natural interpretations.
181. In the next portion of the lecture, we'll talk about how we can use policy evaluation in a complete Actor-Critic algorithm to derive a new reinforcement learning method.