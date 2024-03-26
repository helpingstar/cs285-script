1. All right, so far we talked about how we can learn value functions represented in a tabular form.
2. So there's no neural net, no d-bar L, no function approximation, just a small discrete state space where we can enumerate the value in every possible state.
3. Now let's talk about how we can introduce neural networks and function approximation.
4. So first, how do we represent V ?
5. Well, so far we're talking about how we can represent it as a big table with one entry for each discrete state.
6. So in state 0, you say V is 0.2, in state S equals 1, it's 0.3, and so on.
7. Why is this not enough?
8. Well, take a moment to think about this.
9. Why might we not want to represent the value function as a table?
10. Let's say that you're playing a video game from images, right.
11. Now in this video game, the number of possible states, if you have a 200x200 picture, is 255, which is the number of values that each pixel can take on, raised to the third power because there are three color channels, raised to the power of 200x200.
12. So these are, this is the number of possible images you can see.
13. Maintaining a table over this many entries is impossible, this is more than the number of atoms in the universe, and that's for a discrete state space.
14. For a continuous state space, it's actually just infinite, and this would never be possible.
15. This is also, by the way, sometimes referred to as the curse of dimensionality.
16. So if you have a table that has a number of entries, you can say that you have a table that has a number of entries, and that's the curse of dimensionality.
17. If someone says curse of dimensionality in the context of reinforcement learning, what that refers to is the simple fact that if you have a state space with many dimensions, the number of entries that you need in a table for tabular reinforcement learning is exponential in the number of dimensions.
18. So we'll use a function approximator.
19. Let's say just like in lecture 6, we're going to have a neural net value function that maps from states to scalar valued values.
20. So we're going to have a neural net that looks like this, and it has some parameters, fine.
21. So what we can do is we can fit our neural net value function in much the same way as we did in lecture 6 by doing least squares regression onto target values.
22. And if we use the value iteration procedure from the previous section, then our target values are just the max over a of q π sa.
23. So the 38.
24. Jerπ is going to be in theCu and traqn.
25. So for the list of operations we could take in too complete an account, we could take as too large an account as people did, we can just stake out a maximum of 1 ут each time taking that capital and having it careers았어요 a maximum of 1 ут would look like this.
26. Then our fitted value iteration algorithm would look like this.
27. We would take our dataset of states and actions.
28. For every sample state, we would evaluate every possible action you could take in that state, and we would take a max over those actions of our q values.
29. So what I have in the parentheses here is I have just substituted in the q value.
30. So the q values of the reward, plus γ times the expected value of the next state.
31. You'll see this is real.
32. Here is ourginqu.tm.
33. Now we get 5u.
34. Let's take one字 speed our second the next state.
35. So I've substituted that into the parameters.
36. We don't represent the Q function explicitly here.
37. We just compute it as we go to evaluate the max.
38. And that gives us our target values y.
39. And then we solve for ϕ by doing least squares regression so that v ϕ of si is close to yi.
40. So this is fitted value direction.
41. Step one, compute your target values by constructing the Q function for every possible action at each sampled state.
42. So you have a finite sampling of states and we still assume that we have discrete action space so we can perform this enumeration exactly.
43. For every action we valued its reward plus γ times the expected value of the value at the next state.
44. Do the max over that.
45. That gives us our target value.
46. And then in step two, regress onto those target values.
47. All right.
48. So this is a reasonable algorithm that we could use.
49. But it still requires us to know the transition dynamics.
50. Where do we need to know the transition dynamics?
51. Which part of this algorithm requires knowing the transition dynamics?
52. Well, it's basically this part.
53. So there are two ways in which this requires knowledge of the transition dynamics.
54. It requires being able to compute that expected value.
55. And perhaps more importantly, it requires us to be able to try multiple different actions from the same state, which we can't do in general if we can only run policies in the environment, instead of teleporting to a state multiple times and trying multiple actions from the same exact state.
56. So if we don't know the transition dynamics, generally we can't do this.
57. So let's go back to policy iteration.
58. In policy iteration, we alternated between evaluating q π or a π.
59. But if you have a π, or if you have q π, you can recover a π.
60. And then step two, setting our policy to be this greedy arc max policy.
61. So that was policy iteration.
62. And step one in policy iteration involved policy evaluation, which involved repeatedly applying this value function recurrence that we saw before.
63. So what if instead of applying the value function recurrence to learn the value function, we instead directly constructed a Q function recurrence in an analogous way?
64. So if I wanted to construct the Q function at a particular state action tuple, I can write exactly the same recurrence.
65. Except that now, since the Q function is a function of a state and an action, I don't need to evaluate the next state given s and π of s.
66. I just evaluate the next state given the sa tuple that I'm training my Q function on.
67. And this might at first seem like a very subtle difference, but it's a very, very important one.
68. Because now, as my policy π changes, the action for which I need to sample s prime, basically the a that's on the right of the conditioning bar and p of s prime given sa, doesn't actually change.
69. Which means that if I have a bunch of samples, s comma a comma s prime, I can use those samples to fit my Q function regardless of what policy I have.
70. The only place where the policy shows up is as an argument to the Q function at the state s prime inside of the expectation.
71. And it turns out that this very seemingly very simple change allows us to perform policy iteration style algorithms without actually knowing the transition dynamics, just by sampling some s, a, s prime tuples, which we can get by running any policy we want.
72. So, this second recurrence that I've written here doesn't require knowing the transition probabilities, it just requires samples of the form s comma a comma s prime.
73. So if we do this for step one in policy iteration, we would no longer require knowing the transition probabilities.
74. And this is very important.
75. This is the basis of most value-based model-free RL algorithms.
76. All right.
77. Now, we seemingly took a step back because before we derived policy iteration and then we simplified it to get value iteration.
78. And the way that we got value iteration is by using this max trick.
79. In value iteration, we saw that when we construct the policy, we take the argmax, but then we simply take the value of that argmax action.
80. So evaluating the value of the argmax is just like taking the max.
81. So we can forego policy construction.
82. We can forgo that step two and directly perform value iteration.
83. Can we do the same max trick with Q functions?
84. So can we essentially do something like value iteration, but without knowing the transition?
85. So what we did before is we took policy iteration, which alternates between evaluating the value function in step two and setting the policy to be the greedy policy in step, sorry, evaluating the value function in step one and setting the policy to be the greedy policy in step two.
86. And we transformed to this other algorithm where step one constructs target values by taking a max over the Q values.
87. And step two fits a new value function to those target values.
88. So the way that we construct a fitted Q iteration algorithm is very much analogous to fitted value iteration.
89. We construct our target value YI as the reward of a sampled state action tuple SI AI plus γ times the expected value of the value function at state s'.
90. and then in step 2 we simply regress our Q function Q ϕ onto those target values.
91. The trick, of course, is that we have to evaluate step 1 without knowing the transition probabilities, so we're going to do two things.
92. First, we're going to replace V of SI prime with the max over A at Q ϕ SI prime AI prime, because we're only approximating Q ϕ, we're not approximating V ϕ.
93. And second, instead of taking a full expectation over all possible next states, we're going to use the sampled state SI prime that we got when we generated that sample.
94. And now all we need to run this fitted Q iteration algorithm is samples SI AI SI prime, which can be constructed by rolling out our policy.
95. So this is fitted Q iteration.
96. It alternates between two steps.
97. Step one, estimate target values, which you can do using only samples, and step two, estimate target values, which you can do using only samples.
98. And your previous Q function Q ϕ.
99. Step two, fit a new ϕ with regression onto your target values using the same exact samples that you used to compute your target values.
100. And this doesn't require simulation of different actions.
101. It only requires the actions that you actually sampled last time when you ran your policy.
102. So this works even for off-policy samples.
103. So this algorithm does not make any assumptions that the actions were actually sampled.
104. So this algorithm does not make any assumptions that the actions were actually sampled from the latest policy.
105. The actions could have been sampled from anything.
106. So you can store all the data you've collected so far.
107. It doesn't need to come from your latest policy, unlike actor critic, where we had a non-policy algorithm.
108. There's only one network.
109. There's no policy gradient at all.
110. There's no actor.
111. There's only a Q function estimator, which is a neural network that takes in a state and an action and outputs a scalar valued Q value.
112. Unfortunately, it turns out that this procedure does not have any convergence guarantees for the action.
113. So we're going to use a scalar valued Q value.
114. So if you do this with a neural net, it may not converge to the true solution, and we'll discuss this a lot more later in the lecture.
115. If you use a tabular representation, it is actually guaranteed to converge.
116. But for a neural network, it's in general not guaranteed to converge.
117. All right.
118. So just to put the pieces together, here's the full fitted Q iteration algorithm.
119. And for each step of the algorithm, there are some free parameters that I'm going to mention.
120. Step one, we're going to use a non-linear function approximation.
121. So we're going to use a non-linear function approximation.
122. So step one, collect the data set consisting of tuples si, ai, si prime, and ri using some policy.
123. The algorithm works for a wide range of different policies.
124. Now, not all policy choices are equally good, but the principles will apply to any policy, and it certainly doesn't have to be the latest policy.
125. And one of the parameters you have to choose is the number of such transitions you are to collect.
126. So typically, you would draw your policy for some number of steps or some number of steps.
127. So you would draw your policy for some number of steps or some number of trajectories, but you get to choose how many.
128. And of course, you also choose the policy that you're going to be rolling out.
129. What policy do you use to collect this data?
130. A very common choice is, in fact, to use the latest policy, but there are a few nuances about that choice that I'll discuss shortly.
131. Step two, for every transition that you sampled, calculate a target value.
132. So you calculate the target value, yi, by taking the reward from that transition, plus γ times the max over the next action, ai prime, of the Q value, Q ϕ, si prime, ai prime, using your previous Q function estimator, Q ϕ.
133. Step three, train a new Q function, which means find a new parameter vector ϕ, by minimizing the difference between the values of Q ϕ, si, ai, and the corresponding target value, yi.
134. So you have a Q function, which takes as input s and a.
135. It outputs a scalar value, and it has parameters ϕ.
136. I should mention, by the way, that a very common design for a neural network architecture for a Q function with discrete actions is actually to have the actions be outputs rather than inputs.
137. So an alternative design is to input the state s, and then output a different Q value for every possible action a.
138. You can think of that as a special case of this design, and I'll discuss in class a little about how those relate.
139. But conceptually it's probably easiest to think about is a neural network that takes s and a as input, and outputs a value.
140. But you could also think of it as a network that takes s as input, and outputs a different value for every possible a.
141. So in step three, one parameter you have to choose is the number of gradients steps, capital S, that you will make in performing this optimization.
142. You can run this optimization all the way convergence or you can run it for just a few gradient steps.
143. Now doing step 3 once doesn't actually get you the best possible Q function.
144. You could alternate step 2 and step 3 some number of times, let's say capital K times, before going out and collecting more data.
145. And the number of times you alternate step 2 and step 3 we're going to refer to as K, that's the number of iterations of the fit-or-q iteration that you take in the inner loop.
146. And then once you've taken those K iterations maybe you could take your latest policy, modify it with some exploration rules, which I'll discuss shortly, and use it to collect some more data.
147. So this is the general design of fit-or-q iteration.
148. Many different algorithms can actually be interpreted as variants of fit-or-q iteration, including algorithms like Q-learning, which I will cover shortly.
149. Alright, so to review this portion of the lecture, we discussed value-based methods, and we'll talk about the different ways that we can use fit-or-q to generate value-based methods.
150. Value-based methods do not learn a policy explicitly, they just learn a value function or a Q function represented as a table or a neural network.
151. If we have a value function, we can recover a policy by using the argmax policy.
152. We talked about how fit-or-q iteration removes the need for us to need to know the transition probabilities, and we discussed this kind of generic form of the fit-or-q iteration algorithm.