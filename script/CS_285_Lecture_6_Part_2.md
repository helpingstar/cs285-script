1. All right, now that we've talked about policy evaluation and how value functions can be incorporated into the policy gradient, let's put these pieces together and construct an Actor-Critic reinforcement learning algorithm.
2. So a basic batch Actor-Critic algorithm can look something like this.
3. This is kind of based on the REINFORCE procedure before, but with some additional steps added.
4. So step one, just like before, is going to be to generate samples by running rollouts through our policy.
5. That's basically the orange box.
6. And that remains essentially unchanged.
7. Step two is to fit our approximate value function to those sampled rewards.
8. And that's what's going to replace the green box.
9. So instead of just naively summing up all the rewards, we're now going to fit a neural network as we discussed in the previous section.
10. Step three, for every state action tuple that we sampled, evaluate the approximate advantage as the reward plus the approximate value of the next state minus the value of the current state.
11. Step four, use these advantage values to construct a policy gradient estimator by taking ∇log π at every time step and multiplying it by the approximate advantage.
12. And then step five, like before, is to take a gradient ascent step.
13. So the part that we talked about when we discussed policy evaluation is mostly the step two how do we could actually fit the value function.
14. And we talked about how we can make a number of different choices.
15. We could fit it to single sample Monte Carlo estimates, meaning that we actually sum up the rewards that we got along that trajectory.
16. And that gives us our target values.
17. We also talked about how we could use bootstrap estimates, where we use the actual observed reward plus the estimated value at the next state by using our previous value function estimator.
18. And these give us a few different options how to fit the critic.
19. Now, at this point, I want to make a little aside to discuss what happens when we fit value functions with this bootstrap rule in infinite horizon settings.
20. So the trouble that we might get into is if the episode length is infinite, then each time we apply this bootstrap rule, our value function will increase.
21. So if we have, for example, an episodic task that ends at a fixed time.
22. Maybe this is not such a big issue.
23. Perhaps we could have a different value function for every time step and everything is finite horizon.
24. Episodic tasks are fairly common in some settings, like this robotic task here.
25. But we could have an infinite horizon, a continuous or cyclic task, like this running task.
26. Or we might simply want to use the same value function for all time steps.
27. In these cases, using a bootstrap rule the way that I discussed is liable to lead to some problems.
28. For example, if the rewards are always positive each time you bootstrap in this way, your value function increases.
29. And eventually, your value function might become infinite.
30. So how can we modify this rule to ensure that we can always have finite values and that we can handle infinite horizon settings?
31. Well, one very simple trick is to assume that we want a larger reward sooner rather than later.
32. This is very natural.
33. If you imagine that I were to tell you that I'll give you 100,000.
34. You might be quite pleased about that.
35. If I tell you that I'll give you 100,000 next year, you'll probably still be somewhat pleased, but less so.
36. If I tell you that I'll give you 100,000 in a million years, you probably won't take me very seriously.
37. Why?
38. Well, because it matters a lot less to you what will happen in one year and significantly less what will happen in a million years simply because there's so much uncertainty about what will happen to everybody, including you, in that amount of time that those delayed rewards just don't have any value.
39. Another way of thinking about it is that you'd prefer a reward sooner rather than later for a very basic biological reason, which is that someday you're going to die and you'd like to receive the reward before you die.
40. And if I tell you you'll get the reward in a million years, then it's very unlikely that you'll get the reward before you die.
41. That might sound kind of grim, but we can use this cute metaphor to actually construct a solution to this infinite reward problem.
42. We can favor a reward sooner rather than later by actually modeling the fact that the agent might, quote unquote, die.
43. So the way that we're going to do this is we will introduce a little multiplier in front of the value.
44. So instead of setting the target value to be r plus the next V, we'll set it to be r plus the next V times γ, where γ is what we call a discount factor.
45. It's a number between 0 and 1.
47. 0.99 works really well if you want an example of a discount factor.
48. Generally, we choose them to be somewhere between 0.9 and 0.999.
49. And one way that you can interpret the role of γ is that γ kind of changes your MDP.
50. So let's say that you have this MDP, where you have four states and you can transition between those four states.
51. And those transitions are governed by some probability distribution, p(s'|s,a).
52. When we add γ, one way we can think about this is that we're adding a fifth state, a death state.
53. And we have a probability of (1-γ) of transitioning to that death state at every time step.
54. Once we enter the death state, we never leave.
55. So there's no resurrection in this MDP.
56. And our reward is always 0.
57. So that means that the expected value for the next time step will always be expressed as γ times its expected value in the original MDP plus (1-γ) times 0.
58. And that's where we get this γ factor.
59. So the probability of entering the death state is (1-γ), which means that the modified dynamics now are just γ times the original probabilities.
60. And that (1-γ) remaining slice accounts for entering the death state.
61. So mechanically, the modification that we have with the discount factors just multiply our values by γ for every time step that we back them up.
62. And what that does is it makes us prefer rewards that happen sooner rather than later.
63. And mathematically, one way that we can interpret this is that we're actually modifying our MDP to introduce the probability of death with probability of 1 minus gamma.
64. All right.
65. Let's dig into discount factors a little bit more.
66. First, could we introduce discount factors into regular Monte Carlo policy gradients?
67. Well, the answer is we most definitely can.
68. So for example, there's one option for how to do this is we can just take that single sample reward-to-go calculation and we can put γ into it.
70. So the equation I have here is exactly the reward-to-go calculation I had before, except that now I've added this γ raised to the power (t' - t) in front of my reward.
71. So that means that the first reward, the one that happens at time step t, has a multiplier of 1.
72. The next one at t plus 1 gets a multiplier of gamma.
73. The next one at t plus 2 gets a multiplier of γ squared and so on.
74. So we're much more affected by rewards that happen closer to us in time.
75. This type of estimator is essentially the single sample version of what you would get if you were to use the value function with the discounted bootstrap.
76. There is another way that we can introduce a discount into the Monte Carlo policy gradient, which seems like it's very similar but has a subtle and important difference.
77. What if we take the original Monte Carlo policy gradient?
78. Well, we can take the original Monte Carlo policy gradient that we had before we did that causality trick, where we just sum together the ∇log π's and then multiply them together with the sum of the rewards.
79. And then we're going to put a discount into that.
80. We'll put a γ^(t-1) multiplier in front of the reward, so that the reward at the first time step is multiplied by 1, the reward at the second time step is multiplied by γ, the reward at the third time step is multiplied by γ squared and so on.
81. Take a moment to think about how these two options compare.
82. Consider if these two options are actually identical mathematically or not.
83. So if we were to apply the causality trick to option 2, meaning that we remove all of the rewards from the past, will we end up with option 1 or not?
84. So we'll come back to this question shortly.
85. But to help us think about how these options compare, let's write out just for completeness what we would get if we had a critic.
86. So with a critic, this is the gradient that we would get.
87. We have the current reward plus γ times the next value minus the current value.
88. And that's our approximate advantage.
89. So option 1 and option 2 are not the same.
90. In fact, option 1 matches the critic version with the exception that we have a single sample estimator.
91. Option 2 does not.
92. In fact, if we were to rewrite option 2 by using the causality trick where we distribute the rewards inside the sum over ∇log π's and then eliminate all of the rewards that happened before the current time step, we'll end up with this expression.
93. We'll end up with ∇log π times step t times the sum from (t'=t) to capital T of γ^(t'-1).
94. Whereas before, we had γ to the (t' - t).
95. So what's going on here?
96. Why do we have this difference?
97. Well, one way that we can understand this difference is if we take the γ^(t-1) factor and distribute it out of the sum.
98. So the last line I have here is exactly equal to the preceding line.
99.  I've just distributed out a γ^(t-1) factor.
100. So now the reward to go calculation is exactly the same as option 1, but I have this additional multiplier of γ^(t-1) in front of ∇log π.
101. So what is that doing?
102. Well, what that's doing is actually quite natural.
103. It's saying that because you have this discount, not only do you care less about rewards further in the future, you also care less about decisions further in the future.
104. So if you're starting at time step 1, rewards in the future matter less, but also your decisions matter less because your decisions further in the future will only influence future rewards.
105. So as a result, you actually discount your gradient at every time step by γ^(t-1).
106. Essentially, it means that making the right decision at the first time step is more important than making the right decision at the second time step because the second time step will not influence the first time step's reward.
107. And that's what that γ^(t-1) factor out front represents.
108. This is in fact the right thing to do if you truly want to solve a discounted problem.
109. If you are really in a setting where you have a discounted factor, and that discount factor represents your preference for near-term rewards or equivalently the probability of entering the death state, then in fact your policy gradient should discount future gradients.
110. Because in a truly discounted setting, making the right decision now is more important than making the right decision later.
111. Coming back to my analogy about the $100, if I tell you that I will give you $100 if you pass my math exam, and I tell you the same thing that I can give you exam in today, or I can give you the exam next year, or I can give you the exam in a million years, well chances are if you know that I'm going to give you the exam in a million years, you're probably not going to study for it.
112. So your policy gradient for that math exam will have a very small multiplier because you'd rather deal with things that will give you rewards much nearer to the present.
113. So it makes sense to have this γ^(t-1) term out front if we're really solving a discounted problem.
114. But in reality, this is often not quite what we want.
115. So saying that later time steps matter less might not actually give us the solution that we're after.
116. So this is the death version.
117. Later steps don't matter if you're dead.
118. It's all mathematically consistent.
119. The version that we actually usually use is option 1.
120. Why is that?
121. Well take a moment to think about that.
122. Why would we prefer to use option 1 instead of option 2?
123. So if we think about this cyclic continuous RL task that I presented before, where the goal is to make this character run as far as possible, while we can model this as a task with discounted reward, in reality, we really do want this guy to run as far as possible, ideally infinitely far.
124. So we don't really want a discounted problem.
125. What we want to do is we want to use the discount to help get us finite values so that we can actually do RL.
126. But then what we'd really like to do is get a solution that works for, you know, for running for arbitrarily long periods of time.
127. So option 1, in some ways, is closer to what we want.
128. Maybe what we really want is more like average reward.
129. So we want to put a 1 over capital T and remove the discount altogether.
130. Average reward is computationally and algorithmically very, very difficult to use.
131. So we would use discount in practice because it's so mathematically convenient, but omit the γ^(t-1) multiplier that shows up in option 2 because we really do want a policy that does the right thing at every time step, not just in the early time steps.
132. Another way to think about the role that the discount factor plays that provides an alternative perspective to this death state, you can read about this in this paper by Philip Thomas called "Bias and Natural Actor-Critic Algorithms", is that the discount factor serves to reduce the variance of your policy gradient.
133. So if you have infinitely large rewards, you also have infinitely large variances, right?
134. Because infinitely large values have infinite variances.
135. By ensuring that your reward sums are finite by putting a discount in front of them, you're also reducing variance at the cost of introducing bias by not accounting for all those rewards in the future.
136. All right.
137. So what happens when we introduce the discount into our Actor-Critic algorithm?
138. Well, the only thing that changes is step 3.
139. So in step 3, you can see that we've added a γ in front of V^π_ϕ(s').
140. Everything else stays exactly the same.
141. One of the things we can do with Actor-Critic algorithms once we take them into the infinite horizon setting is we can actually derive a fully online Actor-Critic method.
142. So we can actually derive a fully online Actor-Critic method.
143. So, so far when we talked about policy gradients, we always used policy gradients in a kind of episodic batch mode setting where we collect a batch of trajectories.
144. Each trajectory runs all the way to the end.
145. And then we use that batch to evaluate our gradient and update our policy.
146. But we could also have an online version when we use Actor-Critic where every single time step, every time we step the simulator or we step the real world, we also update our policy.
147. And here's what an online Actor-Critic algorithm would look like.
148. We would take an action, a, sample from π_θ(a|s), and get a transition (s,a,s',r).
149. So we would take one time step.
150. And at this point I'm not putting t subscripts on anything because this can go on in a single infinitely long non-episodic process.
151. Step two, we update our value function by using the reward plus the value of the next state as our target.
152. Because we're using a bootstrapped update, we don't actually need to know what state we'll get at the following time step or the one after that or the one after that.
153. We just need the one next time step s'.
154. So we don't need s double prime, s triple prime, etc.
155. because we're using the bootstrap.
156. So that's enough for us to update our value function.
157. Step three, we evaluate the advantage as the reward plus the value function of the next state minus the value function of the current state.
158. Again, this only uses things that we already know.
159. It uses (s,a,s',r) and our learned value function.
160. And then using this, we can construct an estimate for the policy gradient by simply taking the ∇log π for this action that we just took multiplied by the advantage that we just calculated.
161. And then we can update the policy parameters with policy gradient.
162. And then we repeat this process.
163. And we do this every single time step.
164. Now there are a few problems with this recipe when we try to do Deep RL.
165. And maybe each of you could take a moment to think about what might go wrong with this algorithm if we implement it in practice.
166. This is kind of the textbook online Actor-Critic algorithm.
167. But for Deep RL, it's a bit problematic.
168. All right.
169. Let's continue this in the next section.