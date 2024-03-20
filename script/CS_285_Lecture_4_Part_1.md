1.  Hello, and welcome to the fourth lecture of CS2E5.
2. In today's lecture, we're going to go over a comprehensive introduction to reinforcement learning algorithms, definitions, and basic concepts.
3. So let's start with some definitions.
4. First, let's go over some of the terminology that we covered in the previous lecture.
5. When we talked about imitation learning, we learned that we can represent a policy as a distribution over actions AT, conditional observations OT.
6. We call this policy pi, and we often use a subscript theta to denote that the policy depends on a vector of parameters that we're going to denote theta.
7. When we're doing deep reinforcement learning, oftentimes we will represent the policy with a deep neural network, although, as we will learn in the next two lectures in the course, depending on the type of reinforcement, learning algorithm, we might choose to represent the policy directly or implicitly through some other object, such as a value function.
8. Important definitions to know are the state, which we denote ST, the observation OT, and the action AT.
9. As we learned in the imitation learning lecture, the observation and state can be related to one another by the following graphical model, where the edge between observations and actions is the policy, the edge between current states and actions and future states, and the position of the state is the transition probability, or the dynamics, and the state satisfies the Markov property, which means that the state at time t plus 1 is independent of the state at time t minus 1, when conditioned on the current state ST.
10. The Markov property is the main thing that distinguishes the state from the observation.
11. The state has to satisfy the Markov property, whereas the observation does not.
12. And we learned in the imitation learning lecture that the observation is some stochastic function of the state, and the observation is the function of the state.
13. The observation is the function of the state, which may or may not contain all the information necessary to infer the full state.
14. So that's kind of the primary difference.
15. We will discuss algorithms for both fully observed reinforcement learning, where we have access to the state, and partially observed reinforcement learning, where you only have access to an observation.
16. Alright, so that's the Markov property.
17. And typically, you'll see me write the policy as πθ AT given OT, or πθ AT given ST, depending on whether I'm talking about the partially observed or the fully observed case.
18. I will sometimes get a little sloppy and use ST, when in fact you could also use OT, but in cases where this distinction is important, I'll make a remark in the lectures.
19. So in imitation learning, we saw that we could collect a dataset, let's say of humans driving a vehicle, consisting of observation action tuples, and then use supervised learning algorithms to figure out how to train a policy to take actions that resemble those of the expert.
20. In today's lecture, we'll introduce the formalism of reinforcement learning, which allows us to train these policies without having access to expert data.
21. So to do that, of course, we need to define what it is that we want the policy to do.
22. And we define the objective by means of something called a reward function.
23. So we could say, well, which action is better or worse?
24. If you're driving this car.
25. If you don't have any data, how can you say what is a good action, what is a bad action?
26. So the reward function essentially tells you that.
27. The reward function is a scalar valued function of the state and the action, although sometimes it can depend on only the state.
28. Most generally, it can depend on both the state and the action.
29. And it tells us which states and actions are better.
30. So for example, if you're trying to drive a car, you could say, well, a state where the car is driving quickly on the road is a high reward state, whereas a state where the car is collided with another car is a low reward state.
31. But crucially, the objective in reinforcement learning is not just to take actions that have high rewards right now, but rather to take actions that will lead to higher rewards later.
32. So if you're driving on the road a little too fast, you might be getting a high reward, but that might lead to an inevitable collision later that will lead to lower reward.
33. So you have to consider the future rewards when choosing the current action.
34. And that's really at the heart of the decision-making problem.
35. That's at the heart of the reinforcement learning problem.
36. How do you choose the right actions now to receive higher rewards later?
37. OK, so together, the state, the action, the reward, and the transition probabilities define what we call a Markov decision process.
38. It is a decision process on a Markovian state.
39. So let's build up towards a full formal definition.
40. We'll start with something called a Markov chain.
41. The Markov chain is named after Andrei Markov, who was a mathematician who pioneered the study of stochastic processes, including Markov chains.
42. And the Markov chain has a very simple definition.
43. It consists of just two things, a set of states, s, and a transition function, t.
44. The state space is simply a set, which could be either discrete or continuous, so you could have a discrete state, in which case each state is a discrete element in a finite-size set, or you could have a continuous state, in which case perhaps your states correspond to real-value vectors in Rn.
45. t is a transition operator.
46. It can also be referred to as a transition probability or a dynamics function.
47. It specifies a conditional probability distribution.
48. So in a Markov chain, t denotes the probability of the state at time t, condition on the state at time t.
49. And the reason that it's called an operator is because if we represent the probabilities of each state at time step t as a vector, so let's say we have n states, this becomes a vector with n elements, and we can call it mu for the probability of the ith state.
50. The whole vector would be called mu .
51. Then we can write the transition probability distribution, or mu .
52. This is called a matrix, where the ijth entry is the probability of going into state i if you are currently in the state j.
53. And if we do this, then we can express the vector of state probabilities at the next time step, mu , as simply a matrix vector product between the matrix of probabilities t and the vector of state probabilities mu .
54. This is simply a way of writing the chain rule of probability distribution.
55. chain rule of probability with a little bit of linear algebra.
56. But here you can see that t acts on mu t as a linear operator, which is why we call it the transition operator.
57. It's an operator that when applied to the current vector of state probabilities produces the next vector of state probabilities.
58. So here's the graphical model corresponding to the Markov chain, and here is the edge denoting transition probabilities.
59. And of course the states in the Markov chain satisfy the Markov property, which means that the state at time t plus 1 is conditionally independent of the state at time t minus 1 given the state at time t.
60. Alright, now the Markov chain by itself doesn't allow us to specify a decision-making problem because there's no notion of actions.
61. So in order to go from the Markov chain to the transition operator, we have to go from the Markov chain to the transition operator.
62. In order to go towards the notion of actions, we need to turn the Markov chain into a Markov decision process, and this was really a much more recent invention pioneered than the 1950s.
63. So the Markov decision process adds a few additional objects to the Markov chain.
64. It adds an action space and a reward function.
65. So now we have a state space, which is a discrete or continuous set of states.
66. We have an action space, which is also a discrete or continuous set.
67. So the graphical model now contains a discrete or both states and actions, and our transition probabilities are now conditional on both states and actions.
68. So we have P given .
69. T is still called a transition operator, but it can no longer be expressed as a matrix, now it's actually a tensor, because it has three dimensions, the next state, the current state, and the current action.
70. But we can do the same kind of linear algebra trick.
71. So if we let denote the probability of being in state j at time t, and we can have another vector that will denote the probability of taking some action, and now we can write t as a tensor, so is the probability of entering state i if you're in state j and taking action k.
72. Then you can write a linear form that describes the state probability mu of the state j.
73. So we can write the next state, comma i, at the next time step as a linear function of the current state probabilities, the current action probabilities, and the transition probabilities.
74. So that means that this transition operator, although it is now a tensor, is still a linear operator that transforms current action and state probabilities into next time step state probabilities.
75. Now we also have this reward function, and the reward function is a mapping from the Cartesian product of the state and action space into real value numbers.
76. And this is what allows us to define an objective for reinforcement learning.
77. So we call r of s, t, comma a, t the reward, and our objective, which I will define in a few slides from now, will be to maximize total rewards.
78. But before I do that, I just want to extend this Markov decision process definition to also define the partially observed Markov decision process, and this is what will allow us to bring in the new, the new, and the new.
79. So we can see that the reward function is the reward function, and the reward function is the reward function.
80. And this is what will allow us to bring in the new, the new, and the new.
81. And this is what will allow us to bring in the new, the new, and the new.
82. notion of observations.
83. So a partially observed Markov decision process further augments the definition with two additional objects, an observation space O and an emission probability, or an observation probability, E.
84. So again, s is the state space, a is an action space, and O is now an observation space.
85. The graphical model now looks the same as it did for the MDP, with the addition that we have these observations O that depend on the state.
86. So we have a transition operator just like before, and now we have an emission probability, a p of O given s , and of course we also have the reward function.
87. The reward function is still mapping from states and actions to real numbers, so the reward function conventionally is defined on states, not on observations.
88. But typically in a partially observed Markov decision process, or POMDP, we would be making decisions based on observations without access to the true states.
89. Alright, now that we've defined the mathematical objects of the Markov chain, the Markov decision process, and the partially observed Markov decision process, let's define an objective for reinforcement learning.
90. So in reinforcement learning, we're going to be learning some object that defines a policy.
91. So for now let's just assume that we learn the policy directly, and we'll see later on how there are some other methods that might represent a policy implicitly.
92. But for now we'll be explicitly learning pi theta, a given s, and theta, a given s.
93. We'll come back to the partially observed case later, for now let's just say that our policy is conditional on s, and theta corresponds to the parameters of the policy.
94. So if the policy is a deep neural net, then theta denotes the parameters of that deep neural net.
95. The state goes into the policy, the action comes out, and then the state and action go into the transition probability, basically the physics that govern the world, which produces the next state.
96. Right?
97. So that's the process that we are controlling.
98. Now in this process we can write down a probability distribution over trajectories.
99. So trajectories are sequences of states and actions, s1, a1, s2, a2, etc, etc, until you get to s, t, a, t.
100. For now we will assume that our control problem is finite horizon, which means that the decision-making task lasts for a fixed number of time steps capital T, and then ends.
101. We will extend this to the infinite horizon setting shortly, but for now we'll write down the finite horizon version because it's quite a bit easier to start with.
102. So if we write down the joint distribution of our states and actions, and here I'm putting the subscript theta on this joint distribution to indicate that it depends on the policy pi theta, we can factorize it by using the chain rule in terms of probability distributions that we've already defined.
103. So we have an initial state distribution P .
104. I sort of brush this under the rug when I define the Markov chain, the MDP and the POMDP, but all of these also have an initial state distribution P .
105. And then we have a product over all time steps of the probability of an action, a, t, given s, t, and the probability of the transition to the next time step, s, t plus 1, given s, t, a, t.
106. Now I said that this is derived from the chain rule of probability, but of course in the chain rule of probability you need to condition on all past variables, but here we are exploiting the Markov property to drop the dependence on s, t minus 1, s, t minus 2, etc., etc., because we know that s, t plus 1 is conditionally independent of s, t minus 1, given s, t.
107. So this is how we can define the trajectory distribution.
108. And for notational brevity I will sometimes write P to denote P through s, t, a, t.
109. So tau is just a shorthand for trajectory, and all it means is a sequence of states and actions.
110. Okay, so having defined the trajectory distribution, we can actually define an objective for reinforcement learning, and we can define that objective as an expected value under the trajectory distribution.
111. So the goal in reinforcement learning is to find the parameters theta that define our policy so as to maximize the expected value of the sum of rewards over the trajectory.
112. So we would like a policy that produces trajectories that have the highest value, and the highest possible rewards in expectation.
113. And the expectation, of course, accounts for the stochasticity of the policy, the transition probabilities, and the initial state distribution.
114. So this is the definition of the reinforcement learning objective that we're going to work with.
115. There are, of course, a few variants on this, and we will derive them over the course of the next few lectures, but this is the most basic version.
116. So at this point I would like all of you to pause and look carefully at the table.
117. And I'm going to give you a few examples of the most basic versions.
118. So at this point I would like all of you to pause and look carefully at the table.
119. And I'm going to give you a few examples of the most basic versions.
120. So at this point I would like all of you to pause and look carefully at the subjective and really make sure that you understand what this means.
121. That you understand what it means to have a sum over rewards, what it means to take their expectation under a trajectory distribution, what a trajectory distribution is, and how it is influenced by our choice of policy parameters theta, which in turn influence the policy pi theta.
122. Because if this part is unclear, then what follows in the remainder of this lecture will be quite hard to follow.
123. So please take a moment to think about this.
124. And if you have any questions about the trajectory distribution, please be sure to write a comment on the video.
125. Alright, let's proceed.
126. So one of the things that we might notice about this factorization of the trajectory distribution is that it actually, although it's defined in terms of the objects that we had in the Markov decision process, it can also be interpreted as a Markov chain.
127. And to interpret this as a Markov chain, we need to know that the distribution of the trajectory distribution is actually a chain.
128. And to interpret this as a Markov chain, we need to know that the distribution of the trajectory distribution is actually a chain.
129. And to interpret this as a Markov chain, we need to know that the distribution of the trajectory distribution is actually a chain.
130. And if this is a Markov chain, we need to define an augmented statespace.
131. So our original statespace is S, but we also have these actions, and the actions make this a Markov decision process.
132. But we know that the action depends on the state based on the policy.
133. But we know that the action depends on the state based on the policy.
134. So pi theta A t given S t allows us to get a distribution of our actions conditioned on states.
135. So what we can do, is we can group this state and action together into a kind of augmented state.
136. And now, our augmented states become like so.
137. Let's talk about how augmented states work together.
138. actually form a Markov chain.
139. So P of st plus 1 comma at plus 1 given st comma at, the transition operator in this augmented Markov chain is simply the product of the transition operator in the MDP and the policy.
140. So this can allow us to define the objective in a slightly different way that will be convenient to use in some of our later derivations.
141. So so far I've defined the objective as an expected value under the trajectory distribution of the sum of rewards.
142. But remember that our distribution actually follows a Markov chain with this augmented space and this transition operator is the product of the MDP transitions and the policy.
143. So we could also write the objective by linearity of expectation as the sum over time of the sum of rewards.
144. So this is the sum of rewards.
145. So we could also write the objective as the sum over time of the MDP transitions and the policy.
146. So we could also write the objective as the sum of the expected values under the state action of marginal in this Markov chain of the reward of that time step.
147. So this is just using linearity of expectation to take the sum out of the expectation so that you have a sum over t of the expectation over tau of R st at.
148. And then since the thing inside the expectation not only depends on st at, we can marginalize all the other variables out and we are left with a sum over the expectation under p theta t.
149. So we could also write the objective as the sum over time of the sum over t of the expectation over tau of R st at.
150. Now this might seem like kind of a useless little mathematical, you know, kind of rewriting of the original objective, but it turns out to be quite useful if we want to extend this to the infinite horizon case.
151. So this marginal p theta st given a t in a finite time Markov chain can be obtained just by marginalizing out all the other time steps.
152. But we can also use this objective to get the infinite horizon case.
153. So what if t equals infinity?
154. Well, okay, the first thing that happens if t equals infinity is your objective might become ill-defined.
155. For example, if your reward is always positive, then you have a sum of an infinite number of positive numbers, which is going to be infinity.
156. So we need some way to make the objective finite, and there are a few ways of doing this.
157. One way of doing this, which I'll use now for convenience, but it's actually not the most common way, is to use what's called the average reward formulation.
158. So you basically take this sum of expected rewards and you divide it by capital T.
159. So basically the average reward over all time steps.
160. Dividing by capital T is a constant, so in general this doesn't change the maximum, but then you can take t to infinity and get a well-defined quantity.
161. Later on we'll learn about something called discounts, which is another way to get a finite number for the infinite horizon.
162. But so making this finite is pretty easy, but let's talk about how we can actually define an infinite horizon objective.
163. So we have our Markov chain from before, and our augmented Markov chain has this transition operator, so that means that we can write the vector st plus one comma at plus one as some linear operator t applied to st comma at, and this is the state action transition operator.
164. And more generally we can skip k time steps ahead and we can say that st plus k at plus k is equal to t to the power k times st at.
165. So one question we could ask is, does the state action marginal, p of st comma at, converge to a stationary distribution, basically converge to a single distribution, as little k goes to infinity?
166. If this is true, that means that we should be able to get a very simple solution, which is to say that we can write the state action marginal, p of st comma at, as a single distribution, and we should be able to write the stationary distribution mu as being equal to t times mu.
167. And under a few technical assumptions, namely ergodicity and the chain being aperiodic, we can actually show that the stationary distribution exists.
168. Intuitively being aperiodic simply means exactly what it sounds like, that the Markov chain is not periodic, and being ergodic means that, roughly speaking, every state can be reached from every other state with non-zero probability.
169. The ergodic assumption is that the state action marginal is not periodic, and the state action marginal is not periodic.
170. So this is important because it prevents a situation where, if you start in one part of the MDP, you might never reach another one.
171. So if this is true, if starting in one part may result in you never reaching another part, then where you start always matters, and the stationary distribution doesn't exist.
172. But if this is not the case, if there's even a slight chance of getting to any state from any other state eventually, then you will have a stationary distribution, provided that it's aperiodic.
173. So the stationary distribution must obey this equation, mu equals t times mu.
174. So the stationary distribution must obey this equation, mu equals t times mu.
175. Because otherwise it's not a stationary distribution.
176. So stationary means it's the same before and after the transition.
177. And if it's the same before and after the transition, then applying t enough times will eventually allow you to reach it.
178. You can solve for the stationary distribution simply by rearranging this equation to see that it is equal to tau minus i times mu equals zero.
179. And remember that mu is a distribution.
180. So it's a vector of numbers that are all positive and sum to one.
181. So one way you can find mu is by finding the eigenvector with eigenvalue one for the matrix defined by t.
182. So mu is eigenvector of t with eigenvalue one.
183. And it always exists under the ergodicity and aperiodicity assumptions.
184. So if we know that if we run this Markov chain forward enough times, eventually it'll settle in one.
185. Eventually it'll settle into mu.
186. That means that as t goes to infinity, this sum of the expectations of the marginals becomes dominated by the stationary distribution terms.
187. So you have some finite number of terms initially that are not in the stationary distribution, mu one, mu two, mu three, etc.
188. Then you have infinitely many terms that are very, very close to the stationary distribution.
189. Which means that once you put in the average reward case, so you're going to find one over t.
190. And then take the limit as t goes to infinity.
191. The limit is basically going to be the expected value of the reward under the stationary distribution.
192. And that allows us to define an objective for reinforcement learning in the infinite horizon case as t goes to infinity.
193. Okay, this is perhaps a lot to take in.
194. So this would be a good place to pause, think about the derivation on this slide.
195. And if something is unclear or you have any questions, please be sure to write them in the comments.
196. Alright, now one last bit that I want to describe in this section, which is very important for understanding the basic principle behind a lot of reinforcement learning methods, is that reinforcement learning is really about optimizing expectations.
197. So although we talk about reinforcement learning in terms of choosing actions that lead to high rewards, we're always really concerned about expected values of rewards.
198. And the interesting thing about expected values, is that we're always concerned about expected values of rewards.
199. And the interesting thing about expected values, is that expected values can be continuous in the parameters of the corresponding distributions, even when the function that we're taking the expectation of is itself highly discontinuous.
200. And this is a really important fact for understanding why reinforcement learning algorithms can use smooth optimization methods like gradient descent to optimize objectives that are seemingly non-differentiable, like binary rewards for winning or losing a game.
201. Let me explain this with a little toy example.
202. Let's imagine that you're driving down a mountain road, and your reward is plus one if you stay on the road, and zero if you fall, or negative one if you fall off the road.
203. So the reward function here appears to be discontinuous.
204. There is a discontinuity between staying on the road and falling off the road.
205. And if you try to optimize the reward function with respect to, for example, the position of the car, that optimization problem can't really be solved with gradient-based methods, because the reward is not a continuous, or much less a differentiable function, of the car's position.
206. However, if you have a probability distribution over some action, let's say that abstractly that you just get to choose like fall or don't fall, so you have a binary action, you either fall or you don't fall, and it's a Bernoulli random variable with parameter theta.
207. So with probability theta you fall off, with probability one minus theta you don't fall off.
208. Now, the interesting thing is that the expected value of the reward with respect to pi theta is actually smooth in theta, because you have a probability of theta falling off, which has a reward of minus one, and a probability of one minus theta of staying on the road, so the reward is one minus theta minus theta.
209. And that's perfectly smooth and perfectly differentiable, in theta.
210. So, this is a very important property that will come up again and again, and that it really explains why reinforcement learning algorithms can optimize seemingly non-smooth and even sparse reward functions, which is that expected values of non-smooth and non-differentiable functions under differentiable and smooth probability distributions are themselves smooth and differentiable.
211. Okay, let's pause there.
212. Thank you.