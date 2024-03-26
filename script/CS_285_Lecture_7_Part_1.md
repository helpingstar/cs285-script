1. All right, welcome to lecture 7 of CS285.
2. Today, we're going to talk about value function methods.
3. So, we first saw algorithms that use value functions when we discussed Actor-Critic algorithms.
4. And just to recap, the basic batch mode Actor-Critic algorithm that we discussed extends the policy gradient algorithm to introduce a value function.
5. So, in the Actor-Critic algorithm that we covered in the last lecture, we would generate some samples from our current policy by running that policy on the robot.
6. We would fit a value function to those samples, which is a neural network in the previous lecture that mapped states to scalar-valued values.
7. Then we would use that value function to estimate the advantage for each state action tuple, SI, AI, that we sampled.
8. And we generated these advantage estimates by taking the current reward.
9. Plus the next value minus the current value.
10. And we could also optionally insert a discount factor in front of the next value.
11. So, that's line 3.
12. And then we would use these estimated advantages to estimate a policy gradient on line 4 using the same policy gradient formula that we learned about in the preceding lecture.
13. And then we would do gradient descent on the policy parameters.
14. So, in this scheme, it again follows the usual recipe for a reinforcement learning algorithm that we discussed.
15. So, the orange box.
16. Consists of gerating samples.
17. But the green box.
18. Consists of fitting our value function.
19. and the blue box.
20. Consists of taking a gradient descent step on the policy parameters.
21. So, can we omit the policy gradient entirely?
22. What if we just learn a value function and then try to use that value function to figure out how to act?
23. The intuition for why this should be possible is that the value function tells us which states are better than which other states.
24. we simply select actions that go into the better states, maybe we don't need an explicit policy neural network anymore.
25. So here's the way to make this intuition a bit more formal.
26. a π st at is our advantage.
27. That's the difference between our q value and our value.
28. And intuitively the advantage says how much better is at than the average action according to the policy π, where π is the policy for which we calculated this advantage.
29. So then arg max with respect to at of a π st at is the best action that we could take from st if we follow π thereafter.
30. Which means that the arg max with respect to at of the advantage is going to be at least as good as an action that we would have sampled from our current policy.
31. We know it's at least as good because it's actually the best.
32. So if it's the best action from st if we then follow π thereafter, then it's at least as good as whatever action π would've chosen.
33. And the interesting thing is that this is true regardless of what π actually is.
34. So this means that this arg max should immediately suggest to us that regardless of which policy we had before, even if it was a very bad random policy, we ought to be able to improve it by selecting it.
35. the action according to the argmax of the advantage.
36. So maybe we could forget about representing policies explicitly and we can just use this argmax to select our actions.
37. And that's the basis for value-based methods.
38. So we will construct new policies implicitly, so at every iteration we can construct a new policy pi-prime that assigns a probability of 1 to the action a if it is the argmax of the advantage a , where a is the advantage for the previous implicit policy.
39. Crucially we don't need another neural network to represent this policy.
40. The policy is represented implicitly as this argmax, so the only thing that we need to actually learn is the advantage.
41. And then we will of course re-estimate the advantage function for pi-prime and then construct a new policy that's the argmax with respect to that.
42. So each time we create this implicit pi-prime, we know that that is at least as good as π and in most cases better.
43. So we still have an algorithm with the usual three boxes, where in the orange box are generated samples, in the green box we're going to fit some kind of value function, either q-pi or v-π, which we will use to estimate the advantage, and in the green box we're going to fit some kind of value function, either q-pi or v-π, and in the green box we're going to fit some kind of value function, either q-pi or v-π, and in the blue box, instead of taking a gradient ascent step on an explicit policy, we will construct this implicit policy as the argmax.
44. So there's no actual learning that happens in the blue box anymore, it's just setting the policy to be this argmax policy.
45. So this is the high-level idea behind what is called policy iteration.
46. So in policy iteration, step one is to evaluate the advantage of your current policy π, and then step two is to construct a new policy that's going to be this π prime, where π prime takes an action with probability one if it is the argmax of the advantage.
47. And then we alternate these two steps.
48. It's called policy iteration because we iterate between evaluating the policy in step one and updating policy in step two.
49. So step two is pretty straightforward, especially if we have a discrete action space, computing an argmax is something that we can do in the same way.
50. So we can do this in the same way that we do in the same way that we do in the same way.
51. So step one is to evaluate the advantage of π, and step two is to check the advantage value of π.
52. So step two is to check the advantage value of every possible action.
53. If you have continuous valued actions, things get a little more complex, and we'll cover that case in the subsequent lecture.
54. But for now, let's say that we have discrete actions.
55. The big puzzle is really how to do step one.
56. How do you evaluate the advantage a π for a particular state action tuple for a given previous policy π, which will also be an implicit policy, but we don't care so much about that right now.
57. So like before, we can express the advantage a π sa as the reward sa plus γ times the expected value of v π at s prime minus v π at s.
58. So let's try to evaluate v π of s.
59. One way to evaluate v π of s in order to then estimate these advantages for policy iteration is to use dynamic programming.
60. So for now, let's assume that we have a p of s prime given sa.
61. Let's assume that we know the transition probabilities, and furthermore, let's assume that both s and a are small and discrete.
62. So this is kind of the known dynamic setting.
63. This is not the setting we usually operate in, in model-free RL, but we'll assume that that's our setting for now, just so that we can derive the simple dynamic programming algorithm and then turn it into a model-free algorithm later.
64. So if we have a small discrete s and a, we could imagine that we can essentially enumerate these values and then do our different processes.
65. So when we do this, if we take the first step, we can now estimate the value of π for the transition our entire state and action space.
66. We can represent it with a table.
67. For instance, you might have this grid world.
68. In this grid world, your actions correspond to steps that move left, right, up and down.
69. So here you have 16 states and you have four actions per state, actions for moving left, right, up and down.
70. So in this kind of small state space, you can actually store the full value function v π in a table, right.
71. You can actually construct a table with 16 numbers and just write down the v π for every one of those 16 numbers.
72. You don't need a neural network for that.
73. So here's a potential table of 16 numbers.
74. And your transition probabilities t are represented by a 16 by 16 by 4 tensor.
75. So when we say we're doing tabular reinforcement learning or tabular dynamic programming, what we're really referring to is a setting kind of like this.
76. And now we can write down the bootstrapped update for the value function that we saw in lecture 6 in terms of these explicit known probabilities.
77. So if we want to update v π of s, we can set it to be the expected value with respect to the actions a sampled from a policy pine of the reward s a plus γ times the expected value over s prime sampled from p of s prime given s a of v π s prime.
78. And if you have a tabular MDP, meaning you have a small discrete state space, and you know the transition probabilities, this backup can be calculated exactly.
79. So each of the expected values can be computed by summing over all values of that random variable and multiplying the value inside the parentheses by its probability.
80. And then of course, we need to know v π s prime.
81. So we're just going to use our current estimate of the value function for that value.
82. We're going to basically take that number from the table.
83. And then once we calculate a value function v π this way, then we can construct a better policy π prime, as I mentioned before, by assigning probability 1 to the action that is the arc max of the advantage that we obtain from this value function.
84. Now this also means that our policy will be deterministic.
85. So expected values with respect to this π will be pretty easy to calculate.
86. So we can also use this data to calculate the probability of the action.
87. So we can simplify our bootstrap update by removing the expectation with respect to π and just directly plugging in the only action that has non-zero probability.
88. So then we get the simplified backup where v prime of s is set to r of s comma π of s plus γ times the expectation under p s prime given s π of s of v π s prime.
89. Okay, so now we can, I think, we can do a little bit more awhile over here because I can't see the numbers yet.
90. So first step is we plug this procedure into our policy iteration algorithm.
91. So as a reminder our policy iteration algorithm.
92. Step one is evaluate our advantage, which we can obtain from the value function.
93. So it's really...
94. step one is evaluate the value function.
95. And then step two, set your new policy to be this π prime policy obtained via the arc max.
96. And then repeat.
97. So this is exactly the policy iteration algorithm we had before.
98. And the thing that we're going to learn now is the value function, which for them, for now, in this tabular form as a table of 16 numbers if you have 16 states.
99. So policy evaluation is what goes into step one.
100. And the way that we can do policy evaluation is by repeatedly applying this recursion, by repeatedly setting the value for every state, for every entry in our table, to be the reward at that state plus the expected value of the value at the next state.
101. And we just repeat this multiple times.
102. You can prove that repeating this recursion eventually converges to a fixed point, and this fixed point is the true value function v π.
103. For those of you that are a bit more mathematically inclined, I will also point out that if you write v π of s equals r of s π s plus this expectation, that actually represents a system of linear equations that describe the value function v π.
104. And the system of linear equations, can then be solved with any linear equation solver.
105. So something that you could do as a homework assignment to understand this a little bit better is to actually write down the system of linear equations and work out its solution.
106. It's fairly straightforward to do, but it's a good exercise to make sure that you really understand dynamic programming and policy evaluation.
107. Okay, so we have our tabular MDP, 16 states, 4 actions per state.
108. We can store the full value function in a table.
109. We can compute the value function using policy evaluation by repeatedly using this recursion.
110. And we perform this in the inner loop of our policy iteration procedure, which simply alternates between policy evaluation and updating the policy to be this argmax policy, where the advantage is obtained from the value function that we found in step 1.
111. Now there is an even simpler dynamic programming process that you can design that kind of short circuits this policy iteration procedure.
112. So to see this, here are the following steps that we need.
113. So first notice that we're taking the argmax of the advantage function when you compute the policy.
114. And the advantage is the reward plus the expected next value minus the current value.
115. Now, if you remove the minus v π s, you just get the Q function.
116. Since you're taking the argmax with respect to a, any term that doesn't depend on a actually doesn't influence the argmax.
117. So the argmax of the advantage is actually equal to the argmax of the Q function.
118. So we can equivalently write the new policy as the argmax of q, which is a little simpler because we removed one of the terms.
119. And the way that we can think about this graphically is that the Q function is a table with one entry for every state and every action.
120. So here different rows are different states and different columns are different actions.
121. And when we compute the argmax, we're basically finding the entry in each row that has the largest value.
122. And we're selecting the corresponding index as our policy.
123. When we then later on go on to actually evaluate that policy, we're going to plug that index back into a Q function to get its value.
124. So the argmax gives us the policy, but the max actually gives us the new value of that policy.
125. So what we can do is we can short circuit this.
126. We can actually skip the step where we recover the indices and just directly take the values.
127. So we can skip the policy and compute the values directly.
128. And this gives us a new algorithm, which is called value iteration, where in step one we set the q values, basically the entries in this s by a table, to be the reward plus the expected value of the value function of the next time step.
129. And then in step two we set the value function to be the max over a in this Q function table.
130. So we basically take each row in the Q function table and pick the entry with the largest number in it and store that as the value for that state.
131. So here explicit policy computation is skipped.
132. We don't have our action to present the policy explicitly, but you can think of it as showing up implicitly in step two, because setting the value to be the max over the actions in the q value table is analogous to taking the argmax and then plugging the index of the argmax into the table to recover the value.
133. But since taking the argmax and then plugging into the table is the same as just taking the max, we can basically short circuit that step and get this procedure.
134. So step one, construct your q value table by setting it to be the reward plus the expected value of the next time step.
135. Step two, set the value to be the max.
136. So this yield, slightly modified and simpler procedure, where in the green box you construct your table of q values and the blue box you construct the value function by taking the max.
137. Now this procedure can be simplified even further if you actually take step two and plug it into step one.
138. So you notice that V of s only shows up in one place, which is inside that expectation step one.
139. So if you simply replace that with a max over a of q sa, you don't even need to represent the value function.
140. You only need to represent the Q function.