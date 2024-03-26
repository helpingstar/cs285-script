1. Hello, and welcome to the fourth lecture of CS285.
2. In today's lecture, we're going to go over a comprehensive introduction to reinforcement learning algorithms, definitions, and basic concepts.
3. So let's start with some definitions.
4. First, let's go over some of the terminology that we covered in the previous lecture.
5. When we talked about imitation learning, we learned that we can represent a policy as a distribution over actions a_t, conditional observations o_t.
6. We call this policy π, and we often use a subscript θ to denote that the policy depends on a vector of parameters that we're going to denote θ.
7. When we're doing deep reinforcement learning, oftentimes we will represent the policy with a deep neural network, although, as we will learn in the next two lectures in the course, depending on the type of reinforcement, learning algorithm, we might choose to represent the policy directly or implicitly through some other object, such as a value function.
8. Important definitions to know are the state, which we denote s_t, the observation o_t, and the action a_t.
9. As we learned in the imitation learning lecture, the observation and state can be related to one another by the following graphical model, where the edge between observations and actions is the policy, the edge between current states and actions and future states is the transition probability, or the dynamics, and the state satisfies the Markov property, which means that the state at time t plus 1 is independent of the state at time (t-1), when conditioned on the current state s_t.
10. The Markov property is the main thing that distinguishes the state from the observation.
11. The state has to satisfy the Markov property, whereas the observation does not.
12. And we learned in the imitation learning lecture that the observation is some stochastic function of the state which may or may not contain all the information necessary to infer the full state.
13. So that's kind of the primary difference.
14. We will discuss algorithms for both fully observed reinforcement learning, where we have access to the state, and partially observed reinforcement learning, where you only have access to an observation.
15. Alright, so that's the Markov property.
16. And typically, you'll see me write the policy as π_θ(a_t|o_t), or π_θ(a_t|s_t), depending on whether I'm talking about the partially observed or the fully observed case.
17. I will sometimes get a little sloppy and use s_t, when in fact you could also use o_t, but in cases where this distinction is important, I'll make a remark in the lectures.
18. So in imitation learning, we saw that we could collect a dataset, let's say of humans driving a vehicle, consisting of observation action tuples, and then use supervised learning algorithms to figure out how to train a policy to take actions that resemble those of the expert.
19. In today's lecture, we'll introduce the formalism of reinforcement learning, which allows us to train these policies without having access to expert data.
20. So to do that, of course, we need to define what it is that we want the policy to do.
21. And we define the objective by means of something called a reward function.
22. So we could say, well, which action is better or worse?
23. If you're driving this car.
24. If you don't have any data, how can you say what is a good action, what is a bad action?
25. So the reward function essentially tells you that.
26. The reward function is a scalar valued function of the state and the action, although sometimes it can depend on only the state.
27. Most generally, it can depend on both the state and the action.
28. And it tells us which states and actions are better.
29. So for example, if you're trying to drive a car, you could say, well, a state where the car is driving quickly on the road is a high reward state, whereas a state where the car is collided with another car is a low reward state.
30. But crucially, the objective in reinforcement learning is not just to take actions that have high rewards right now, but rather to take actions that will lead to higher rewards later.
31. So if you're driving on the road a little too fast, you might be getting a high reward, but that might lead to an inevitable collision later that will lead to lower reward.
32. So you have to consider the future rewards when choosing the current action.
33. And that's really at the heart of the decision-making problem.
34. That's at the heart of the reinforcement learning problem.
35. How do you choose the right actions now to receive higher rewards later?
36. OK, so together, the state, the action, the reward, and the transition probabilities define what we call a Markov decision process.
37. It is a decision process on a Markovian state.
38. So let's build up towards a full formal definition of Markov Deicison Processes.
39. We'll start with something called a Markov chain.
40. The Markov chain is named after Andrei Markov, who was a mathematician who pioneered the study of stochastic processes, including Markov chains.
41. And the Markov chain has a very simple definition.
42. It consists of just two things, a set of states, s, and a transition function, t.
43. The state space is simply a set, which could be either discrete or continuous, so you could have a discrete state, in which case each state is a discrete element in a finite-size set, or you could have a continuous state, in which case perhaps your states correspond to real-valued vectors in R^n.
44. T is a transition operator.
45. It can also be referred to as a transition probability or a dynamics function.
46. It specifies a conditional probability distribution.
47. So in a Markov chain, T denotes the probability of the state at time t, condition on the state at time t.
48. And the reason that it's called an operator is because if we represent the probabilities of each state at time step t as a vector, so let's say we have n states, this becomes a vector with n elements, and we can call it μ_{t,i} for the probability of the ith state.
49. The whole vector would be called μ_t.
50. Then we can write the transition probability as a matrix, where the i,jth entry is the probability of going into state i if you are currently in the state j.
51. And if we do this, then we can express the vector of state probabilities at the next time step, μ_{t+1}, as simply a matrix vector product between the matrix of probabilities T and the vector of state probabilities μ_t.
52. This is simply a way of writing the chain rule of probability with a little bit of linear algebra.
53. But here you can see that T acts on μ_t as a linear operator, which is why we call it the transition operator.
54. It's an operator that when applied to the current vector of state probabilities produces the next vector of state probabilities.
55. So here's the graphical model corresponding to the Markov chain, and here is the edge denoting transition probabilities.
56. And of course the states in the Markov chain satisfy the Markov property, which means that the state at time t plus 1 is conditionally independent of the state at time (t-1) given the state at time t.
57. Alright, now the Markov chain by itself doesn't allow us to specify a decision-making problem because there's no notion of actions.
58. So in order to go towards the notion of actions, we need to turn the Markov chain into a Markov decision process, and this was really a much more recent invention pioneered than the 1950s.
59. So the Markov decision process adds a few additional objects to the Markov chain.
60. It adds an action space and a reward function.
61. So now we have a state space, which is a discrete or continuous set of states.
62. We have an action space, which is also a discrete or continuous set.
63. So the graphical model now contains both states and actions, and our transition probabilities are now conditional on both states and actions.
64. So we have p(s_{t+1}|s_t,a t).
65. T is still called a transition operator, but it can no longer be expressed as a matrix, now it's actually a tensor, because it has three dimensions, the next state, the current state, and the current action.
66. But we can do the same kind of linear algebra trick so if we let μ_{t,j} denote the probability of being in state j at time t, and we can have another vector that will denote the probability of taking some action, and now we can write T as a tensor, so T_{i,j,k} is the probability of entering state i if you're in state j and taking action k.
67. Then you can write a linear form that describes the state probability μ_{t+1,i}, at the next time step as a linear function of the current state probabilities, the current action probabilities, and the transition probabilities.
68. So that means that this transition operator, although it is now a tensor, is still a linear operator that transforms current action and state probabilities into next time step state probabilities.
69. Now we also have this reward function, and the reward function is a mapping from the Cartesian product of the state and action space into real value numbers.
70. And this is what allows us to define an objective for reinforcement learning.
71. So we call r(s_t,a_t) the reward, and our objective, which I will define in a few slides from now, will be to maximize total rewards.
72. But before I do that, I just want to extend this Markov decision process definition to also define the partially observed Markov decision process, and this is what will allow us to bring in the notion of observations.
73. So a partially observed Markov decision process further augments the definition with two additional objects, an observation space O and an emission probability, or an observation probability, E.
74. So again, S is the state space, A is an action space, and O is now an observation space.
75. The graphical model now looks the same as it did for the MDP, with the addition that we have these observations o that depend on the state.
76. So we have a transition operator just like before, and now we have an emission probability, a p(o_t|s_t), and of course we also have the reward function.
77. The reward function is still mapping from states and actions to real numbers, so the reward function conventionally is defined on states, not on observations.
78. But typically in a partially observed Markov decision process, or POMDP, we would be making decisions based on observations without access to the true states.
79. Alright, now that we've defined the mathematical objects of the Markov chain, the Markov decision process, and the partially observed Markov decision process, let's define an objective for reinforcement learning.
80. So in reinforcement learning, we're going to be learning some object that defines a policy.
81. So for now let's just assume that we learn the policy directly, and we'll see later on how there are some other methods that might represent a policy implicitly.
82. But for now we'll be explicitly learning π_θ(a|s).
83. We'll come back to the partially observed case later, for now let's just say that our policy is conditional on s, and θ corresponds to the parameters of the policy.
84. So if the policy is a deep neural net, then θ denotes the parameters of that deep neural net.
85. The state goes into the policy, the action comes out, and then the state and action go into the transition probability, basically the physics that govern the world, which produces the next state.
86. Right?
87. So that's the process that we are controlling.
88. Now in this process we can write down a probability distribution over trajectories.
89. So trajectories are sequences of states and actions, s_1, a_1, s_2, a_2, etc, etc, until you get to s_T, a_T.
90. For now we will assume that our control problem is finite horizon, which means that the decision-making task lasts for a fixed number of time steps T, and then ends.
91. We will extend this to the infinite horizon setting shortly, but for now we'll write down the finite horizon version because it's quite a bit easier to start with.
92. So if we write down the joint distribution of our states and actions, and here I'm putting the subscript θ on this joint distribution to indicate that it depends on the policy π_θ, we can factorize it by using the chain rule in terms of probability distributions that we've already defined.
93. So we have an initial state distribution p(s_1).
94. I sort of brush this under the rug when I define the Markov chain, the MDP and the POMDP, but all of these also have an initial state distribution p(s_1).
95. And then we have a product over all time steps of the probability of an action, (a_t | s_t), and the probability of the transition to the next time step, (s_{t+1} | s_t,a_t).
96. Now I said that this is derived from the chain rule of probability, but of course in the chain rule of probability you need to condition on all past variables, but here we are exploiting the Markov property to drop the dependence on s_{t-1}, s_{t-2}, etc., etc., because we know that s_{t+1} is conditionally independent of s_{t-1}, given s_t.
97. So this is how we can define the trajectory distribution.
98. And for notational brevity I will sometimes write p(τ) to denote p of s_1 through s_T, a_T.
99. So τ is just a shorthand for trajectory, and all it means is a sequence of states and actions.
100. Okay, so having defined the trajectory distribution, we can actually define an objective for reinforcement learning, and we can define that objective as an expected value under the trajectory distribution.
101. So the goal in reinforcement learning is to find the parameters θ that define our policy so as to maximize the expected value of the sum of rewards over the trajectory.
102. So we would like a policy that produces trajectories that have the highest possible rewards in expectation.
103. And the expectation, of course, accounts for the stochasticity of the policy, the transition probabilities, and the initial state distribution.
104. So this is the definition of the reinforcement learning objective that we're going to work with.
105. There are, of course, a few variants on this, and we will derive them over the course of the next few lectures, but this is the most basic version.
106. So at this point I would like all of you to pause and look carefully at the subjective and really make sure that you understand what this means.
107. That you understand what it means to have a sum over rewards, what it means to take their expectation under a trajectory distribution, what a trajectory distribution is, and how it is influenced by our choice of policy parameters θ, which in turn influence the policy π_θ.
108. Because if this part is unclear, then what follows in the remainder of this lecture will be quite hard to follow.
109. So please take a moment to think about this.
110. And if you have any questions about the trajectory distribution, please be sure to write a comment on the video.
111. Alright, let's proceed.
112. So one of the things that we might notice about this factorization of the trajectory distribution is that it actually, although it's defined in terms of the objects that we had in the Markov decision process, it can also be interpreted as a Markov chain.
113. And to interpret this as a Markov chain, we need to define a kind of augmented state space.
114. So our original state space is S, but we also have these actions, and the actions make this a Markov decision process.
115. But we know that the action depends on the state based on the policy.
116. So π_θ(a_t|s_t) allows us to get a distribution of our actions conditioned on states.
117. So what we can do, is we can group this state and action together into a kind of augmented state.
118. And now, the augmented states actually form a Markov chain.
119. So p(s_{t+1},a_{t+1}|s_t,a_t), the transition operator in this augmented Markov chain is simply the product of the transition operator in the MDP and the policy.
120. So this can allow us to define the objective in a slightly different way that will be convenient to use in some of our later derivations.
121. So so far I've defined the objective as an expected value under the trajectory distribution of the sum of rewards.
122. But remember that our distribution actually follows a Markov chain with this augmented space and this transition operator is the product of the MDP transitions and the policy.
123. So we could also write the objective by linearity of expectation as the sum over time of the sum of the expected values under the state action of marginal in this Markov chain of the reward of that time step.
124. So this is just using linearity of expectation to take the sum out of the expectation so that you have a sum over t of the expectation over τ of r(s_t,a_t).
125. And then since the thing inside the expectation not only depends on (s_t, a_t), we can marginalize all the other variables out and we are left with a sum over the expectation under p_θ(s_t,a_t) of r(s_t,a_t).
126. So we could also write the objective as the sum over time of the sum over t of the expectation over τ of R s_t, a_t.
127. Now this might seem like kind of a useless little mathematical, you know, kind of rewriting of the original objective, but it turns out to be quite useful if we want to extend this to the infinite horizon case.
128. So this marginal p_θ(s_t|a_t) in a finite time Markov chain can be obtained just by marginalizing out all the other time steps.
129. But we can also use this objective to get the infinite horizon case.
130. So what if T = ∞?
131. Well, okay, the first thing that happens if T = ∞ is your objective might become ill-defined.
132. For example, if your reward is always positive, then you have a sum of an infinite number of positive numbers, which is going to be infinity.
133. So we need some way to make the objective finite, and there are a few ways of doing this.
134. One way of doing this, which I'll use now for convenience, but it's actually not the most common way, is to use what's called the average reward formulation.
135. So you basically take this sum of expected rewards and you divide it by capital T.
136. So basically the average reward over all time steps.
137. Dividing by capital T is a constant, so in general this doesn't change the maximum, but then you can take t to infinity and get a well-defined quantity.
138. Later on we'll learn about something called discounts, which is another way to get a finite number for the infinite horizon case.
139. But so making this finite is pretty easy, but let's talk about how we can actually define an infinite horizon objective.
140. So we have our Markov chain from before, and our augmented Markov chain has this transition operator, so that means that we can write the vector s_{t+1}, a_{t+1} as some linear operator T applied to s_t,a_t, and this is the state action transition operator.
141. And more generally we can skip k time steps ahead and we can say that s{t+k} a{t+k} is equal to T^k times s_t, a_t.
142. So one question we could ask is, does the state action marginal, p(s_t,a_t), converge to a stationary distribution, basically converge to a single distribution, as little k goes to infinity?
143. If this is true, that means that we should be able to write the stationary distribution μ as being equal to Tμ.
144. And under a few technical assumptions, namely ergodicity and the chain being aperiodic, we can actually show that the stationary distribution exists.
145. Intuitively being aperiodic simply means exactly what it sounds like, that the Markov chain is not periodic, and being ergodic means that, roughly speaking, every state can be reached from every other state with non-zero probability.
146. The ergodic assumption is important because it prevents a situation where, if you start in one part of the MDP, you might never reach another one.
147. So if this is true, if starting in one part may result in you never reaching another part, then where you start always matters, and the stationary distribution doesn't exist.
148. But if this is not the case, if there's even a slight chance of getting to any state from any other state eventually, then you will have a stationary distribution, provided that it's aperiodic.
149. So the stationary distribution must obey this equation, μ = Tμ.
150. Because otherwise it's not a stationary distribution.
151. So stationary means it's the same before and after the transition.
152. And if it's the same before and after the transition, then applying t enough times will eventually allow you to reach it.
153. You can solve for the stationary distribution simply by rearranging this equation to see that it is equal to (τ-I)μ = 0.
154. And remember that μ is a distribution.
155. So it's a vector of numbers that are all positive and sum to one.
156. So one way you can find μ is by finding the eigenvector with eigenvalue one for the matrix defined by T.
157. So μ is eigenvector of T with eigenvalue one.
158. And it always exists under the ergodicity and aperiodicity assumptions.
159. So if we know that if we run this Markov chain forward enough times, eventually it'll settle into mu.
160. That means that as t goes to infinity, this sum of the expectations of the marginals becomes dominated by the stationary distribution terms.
161. So you have some finite number of terms initially that are not in the stationary distribution, μ one, μ two, μ three, etc.
162. Then you have infinitely many terms that are very, very close to the stationary distribution.
163. Which means that once you put in the average reward case, so you're going to find 1/T.
164. And then take the limit as T goes to infinity.
165. The limit is basically going to be the expected value of the reward under the stationary distribution.
166. And that allows us to define an objective for reinforcement learning in the infinite horizon case as t goes to infinity.
167. Okay, this is perhaps a lot to take in.
168. So this would be a good place to pause, think about the derivation on this slide.
169. And if something is unclear or you have any questions, please be sure to write them in the comments.
170. Alright, now one last bit that I want to describe in this section, which is very important for understanding the basic principle behind a lot of reinforcement learning methods, is that reinforcement learning is really about optimizing expectations.
171. So although we talk about reinforcement learning in terms of choosing actions that lead to high rewards, we're always really concerned about expected values of rewards.
172. And the interesting thing about expected values, is that expected values can be continuous in the parameters of the corresponding distributions, even when the function that we're taking the expectation of is itself highly discontinuous.
173. And this is a really important fact for understanding why reinforcement learning algorithms can use smooth optimization methods like gradient descent to optimize objectives that are seemingly non-differentiable, like binary rewards for winning or losing a game.
174. Let me explain this with a little toy example.
175. Let's imagine that you're driving down a mountain road, and your reward is plus one if you stay on the road, and zero if you fall, or negative one if you fall off the road.
176. So the reward function here appears to be discontinuous.
177. There is a discontinuity between staying on the road and falling off the road.
178. And if you try to optimize the reward function with respect to, for example, the position of the car, that optimization problem can't really be solved with gradient-based methods, because the reward is not a continuous, or much less a differentiable function, of the car's position.
179. However, if you have a probability distribution over some action, let's say that abstractly that you just get to choose like fall or don't fall, so you have a binary action, you either fall or you don't fall, and it's a Bernoulli random variable with parameter θ.
180. So with probability θ you fall off, with probability one minus θ you don't fall off.
181. Now, the interesting thing is that the expected value of the reward with respect to π_θ is actually smooth in θ, because you have a probability of θ falling off, which has a reward of minus one, and a probability of one minus θ of staying on the road, so the reward is one minus θ minus θ.
182. And that's perfectly smooth and perfectly differentiable, in θ.
183. So, this is a very important property that will come up again and again, and that it really explains why reinforcement learning algorithms can optimize seemingly non-smooth and even sparse reward functions, which is that expected values of non-smooth and non-differentiable functions under differentiable and smooth probability distributions are themselves smooth and differentiable.
184. Okay, let's pause there.