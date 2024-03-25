1. All right, now that we've talked about policy evaluation and how value functions can be incorporated into the policy gradient, let's put these pieces together and construct an Actor-Critic reinforcement learning algorithm.
2. So a basic batch Actor-Critic algorithm can look something like this.
3. This is kind of based on the reinforce procedure before, but with some additional steps added.
4. So step one, just like before, is going to be to generate samples by running rollouts through our policy.
5. That's basically the orange box.
6. And that remains essentially unchanged.
7. Step two is to fit our approximate value function to those sampled rewards.
8. And that's what's going to replace the green box.
9. So instead of just naively summing up all the rewards, we're now going to fit a neural network as we discussed in the previous section.
10. Step three, for every state action tuple that we sampled, evaluate the approximate value of the next state minus the value of the current state.
11. Step four, use these advantage values to construct a policy gradient estimator by taking ∇log π at every time step and multiplying it by the approximate advantage.
12. And then step five, like before, is to take a gradient ascent step.
13. So the part that we talked about when we discussed policy evaluation is mostly the same as the previous step.
14. Right?
15. So we talked about how we could actually fit the value function.
16. And we talked about how we can make a number of different choices.
17. We could fit it to single sample Monte Carlo estimates, meaning that we actually sum up the rewards that we got along that trajectory.
18. And that gives us our target values.
19. We also talked about how we could use bootstrap estimates, where we use the actual observed reward plus the estimated value at the next state by using our previous value function estimator.
20. And these give us a few different options.
21. So let's take a look at how we can fit the value function.
22. So we can see that we have a number of different options for how to fit the critic.
23. Now, at this point, I want to make a little aside to discuss what happens when we fit value functions with this bootstrap rule in infinite horizon settings.
24. So the trouble that we might get into is if the episode length is infinite, then each time we apply this bootstrap rule, our value function will increase.
25. So we can see that we have, for example, an episodic task that ends at a fixed time.
26. Maybe this is not such a big issue.
27. Perhaps we could have a different value function for every time step and everything is finite horizon.
28. Episodic tasks are fairly common in some settings, like this robotic task here.
29. But we could have an infinite horizon, a continuous or cyclic task, like this running task.
30. Or we might simply want to use the same value function for all time steps.
31. In these cases, using a bootstrap rule the way that I discussed is liable to lead to some problems.
32. For example, if the rewards are always positive each time you bootstrap in this way, your value function increases.
33. And eventually, your value function might become infinite.
34. So how can we modify this rule to ensure that we can always have finite values and that we can handle infinite horizon settings?
35. Well, one very simple trick is to assume that we want a larger reward sooner rather than later.
36. This is very natural.
37. If you imagine that I were to tell you that I'll give you 100,000.
38. I would be very happy to give you 100,000.
39. But I would be very unhappy to give you 100,000.
40. You might be quite pleased about that.
41. If I tell you that I'll give you 100,000 next year, you'll probably still be somewhat pleased, but less so.
42. If I tell you that I'll give you 100,000 in a million years, you probably won't take me very seriously.
43. Why?
44. Well, because it matters a lot less to you what will happen in one year and significantly less what will happen in a million years simply because there's so much uncertainty about what will happen to everybody, including you, in that amount of time that those delayed rewards just don't have any value.
45. Another way of thinking about it is that you'd prefer a reward sooner rather than later for a very basic biological reason, which is that someday you're going to die and you'd like to receive the reward before you die.
46. And if I tell you you'll get the reward in a million years, then it's very unlikely that you'll get the reward before you die.
47. That might sound kind of grim, but we can use this cute metaphor to actually construct a solution to this infinite reward problem.
48. We can favor a reward sooner rather than later by actually modeling the fact that the agent might, quote unquote, die.
49. So the way that we're going to do this is we will introduce a little multiplier in front of the value.
50. So instead of setting the target value to be r plus the next v, we'll set it to be r plus the next v times gamma, where gamma is what we call a discount factor.
51. It's a number between 0 and 0.
52. 0 and 1.
53. 0.99 works really well if you want an example of a discount factor.
54. Generally, we choose them to be somewhere between 0.9 and 0.999.
55. And one way that you can interpret the role of gamma is that gamma kind of changes your MDP.
56. So let's say that you have this MDP, where you have four states and you can transition between those four states.
57. And those transitions are governed by some probability distribution, P of S prime given S A.
58. When we add gamma, one way we can think about this is that we're adding a fifth state, a death state.
59. And we have a probability of 1 minus gamma of transitioning to that death state at every time step.
60. Once we enter the death state, we never leave.
61. So there's no resurrection in this MDP.
62. And our reward is always 0.
63. So that means that the expected value for the next time step will always be expressed as gamma times its expected value in the original MDP plus 1 minus gamma times the expected value of gamma times 0.
64. And that's where we get this gamma factor.
65. So the probability of entering the death state is 1 minus gamma, which means that the modified dynamics now are just gamma times the original probabilities.
66. And that 1 minus gamma remaining slice accounts for entering the death state.
67. So mechanically, the modification that we have with the discount factors just multiply our values by gamma for every time step that we back them up.
68. And what that does is it makes us prefer rewards that happen sooner rather than later.
69. And mathematically, one way that we can interpret this is that we're actually modifying our MDP to introduce the probability of death with probability of 1 minus gamma.
70. All right.
71. Let's dig into discount factors a little bit more.
72. First, could we introduce discount factors into regular Monte Carlo policy gradients?
73. Well, the answer is we most definitely can.
74. So for example, there's one option.
75. And the only option for how to do this is we can just take that single sample reward-to-go calculation and we can put gamma into it.
76. So the equation I have here is exactly the reward-to-go calculation I had before, except that now I've added this gamma raised to the power t prime minus t in front of my reward.
77. So that means that the first reward, the one that happens at time step t, has a multiplier of 1.
78. The next one at t plus 1 gets a multiplier of gamma.
79. The next one at t plus 2 gets a multiplier of gamma squared and so on.
80. So we're much more affected by rewards that happen closer to us in time.
81. This type of estimator is essentially the single sample version of what you would get if you were to use the value function with the discounted bootstrap.
82. There is another way that we can introduce a discount into the Monte Carlo policy grading, which seems like it's very similar but has a subtle and important difference.
83. What if we take this value function and we can use it to calculate the value function in the same way that we did in the original Monte Carlo policy grading?
84. Well, we can take the original Monte Carlo policy grading that we had before we did that causality trick, where we just sum together the ∇log pi's and then multiply them together with the sum of the rewards.
85. And then we're going to put a discount into that.
86. We'll put a gamma to the t minus 1 multiplier in front of the reward, so that the reward at the first time step is multiplied by 1, the reward at the second time step is multiplied by gamma, the reward at the third time step is multiplied by gamma squared and so on.
87. Take a moment to think about how these two options compare.
88. Consider if these two options are actually identical mathematically or not.
89. So if we were to apply the causality trick to option 2, meaning that we remove all of the rewards from the past, will we end up with option 1 or not?
90. So we'll come back to this question.
91. We'll come back to this question shortly.
92. But to help us think about how these options compare, let's write out just for completeness what we would get if we had a critic.
93. So with a critic, this is the grading that we would get.
94. We have the current reward plus gamma times the next value minus the current value.
95. And that's our approximate advantage.
96. So option 1 and option 2 are not the same.
97. In fact, option 1 matches the critic version with the exception that we have a single sample estimator.
98. Option 2 does not.
99. In fact, if we were to rewrite option 2 by using the causality trick where we distribute the rewards inside the sum over ∇log pi's and then eliminate all of the rewards that happened before the current time step, we'll end up with this expression.
100. We'll end up with ∇log π times step t times the sum from t prime equals t to capital t of gamma to the t prime minus 1.
101. Whereas before, we had gamma to the t prime minus 1.
102. So before, we had gamma to the t prime minus t.
103. So what's going on here?
104. Why do we have this difference?
105. Well, one way that we can understand this difference is if we take the gamma to the t minus 1 factor and distribute it out of the sum.
106. So the last line I have here is exactly equal to the preceding line.
107. I've just distributed out a gamma to the t minus 1 factor.
108. So now the reward to go calculation is exactly the same as option 1, but I have this additional multiplier of gamma to the t minus 1 in front of ∇log π.
109. So what is that doing?
110. Well, what that's doing is actually quite natural.
111. It's saying that because you have this discount, not only do you care less about rewards further in the future, you also care less about decisions further in the future.
112. So if you're starting at time step 1, rewards in the future matter less, but also your decisions matter less because your decisions further in the future will only influence future rewards.
113. So as a result, you actually discount your gradient at every time step by gamma to the t minus 1.
114. Essentially, it means that making the right decision at the first time step is more important than making the right decision at the second time step because the second time step will not influence the first time step's reward.
115. And that's what that gamma to the t minus 1 factor out front represents.
116. This is in fact the right thing to do if you truly want to solve a discounted problem.
117. If you are really in a setting where you have a discounted problem, and that discount factor represents your preference for near-term rewards or equivalently the probability of entering the death state, then in fact your policy gradient should discount future gradients.
118. Because in a truly discounted setting, making the right decision now is more important than making the right decision later.
119. Coming back to my analogy about the $100, if I tell you that I will give you $100 if you pass my math exam, and I tell you the same thing that I can give you now, or I can give you the exam today, or I can give you the exam next year, or I can give you the exam in a million years, well chances are if you know that I'm going to give you the exam in a million years, you're probably not going to study for it.
120. So your policy gradient for that math exam will have a very small multiplier because you'd rather deal with things that will give you rewards much nearer to the present.
121. So it makes sense to have this gamma to the t minus 1 term out front if we're really solving a discounted problem.
122. But in reality, this is often not quite what we want.
123. So saying that later time steps matter less might not actually give us the solution that we're after.
124. So this is the death version.
125. Later steps don't matter if you're dead.
126. It's all mathematically consistent.
127. The version that we actually usually use is option 1.
128. Why is that?
129. Well take a moment to think about that.
130. Why would we prefer to use option 1 instead of option 2?
131. So if we think about this cyclic continuous RL task that I presented before, where the goal is to make this character run as far as possible, while we can model this as a task with discounted reward, in reality, we really do want this guy to run as far as possible, ideally infinitely far.
132. So we don't really want a discounted problem.
133. What we want to do is we want to use the discount to help get us finite values so that we can actually do RL.
134. But then what we'd really like to do is get a solution that works for, you know, for running for arbitrarily long periods of time.
135. So option 1, in some ways, is closer to what we want.
136. Maybe what we really want is more like average reward.
137. So we want to put a 1 over capital T and remove the discount altogether.
138. Average reward is computationally and algorithmically very, very difficult to use.
139. So we would use discount in practice because it's so mathematically convenient, but omit the gamma to the T minus 1 multiplier that shows up in option 2 because we really do want a policy that does the right thing at every time step, not just in the early time steps.
140. Another way to think about the role that the discount factor plays that provides an alternative perspective to this death state, you can read about this in this paper by Philip Thomas called Bias and Natural Electric Critic Algorithms, is that the discount factor serves to reduce the variance of your policy gradient.
141. So if you have infinitely large rewards, you also have infinitely large variances, right?
142. Because infinitely large values have infinite variances.
143. By ensuring that your reward sums are finite by putting a discount in front of them, you're also reducing variance at the cost of introducing bias by not accounting for all those rewards.
144. So what happens when we introduce the discount into our Actor-Critic algorithm?
145. Well, the only thing that changes is step 3.
146. So in step 3, you can see that we've added a gamma in front of V π ϕ S prime.
147. Everything else stays exactly the same.
148. One of the things we can do with Actor-Critic algorithms once we take them into the infinite horizon setting is we can actually derive a fully online Actor-Critic method.
149. So we can actually derive a fully online Actor-Critic method.
150. So, so far when we talked about policy gradients, we always used policy gradients in a kind of episodic batch mode setting where we collect a batch of trajectories.
151. Each trajectory runs all the way to the end.
152. And then we use that batch to evaluate our gradient and update our policy.
153. But we could also have an online version when we use Actor-Critic where every single time step, every time we step the simulator or we step the real world, we also update our policy.
154. And here's what an online Actor-Critic algorithm would look like.
155. We would take an action, A, sample from π_θ A given S, and get a transition, S comma A comma S prime comma R.
156. So we would take one time step.
157. And at this point I'm not putting T subscripts on anything because this can go on in a single infinitely long non-episodic process.
158. Step two, we update our value function by using the reward plus the value of the next state as our target.
159. Because we're using a bootstrapped update, we don't actually need to know what state we'll get at the following time step or the one after that or the one after that.
160. We just need the one next time step S prime.
161. So we don't need S double prime, S triple prime, etc.
162. because we're using the bootstrap.
163. So that's enough for us to update our value function.
164. Step three, we evaluate the advantage as the reward plus the value function of the next state minus the value function of the current state.
165. Again, this only uses things that we already know.
166. It uses S A S prime R and our learned value function.
167. And then using this, we can construct an estimate for the policy gradient by simply taking the grab log π for this action that we just took multiplied by the advantage that we just calculated.
168. And then we can update the policy parameters with policy gradient.
169. And then we repeat this process.
170. And we do this every single time step.
171. Now there are a few problems with this recipe when we try to do Deep RL.
172. And maybe each of you could take a moment to think about what might go wrong with this algorithm if we implement it in practice.
173. This is kind of the textbook online action-critic algorithm.
174. But for Deep RL, it's a bit problematic.
175. All right.
176. Let's continue this in the next section.