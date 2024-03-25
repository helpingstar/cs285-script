1. The last topic I'll cover in this lecture is a little bit of theory in regard to value-based methods and a little bit more explanation for what I meant before when I said that value-based methods with neural networks don't in general converge to the optimal solution.
2. So to get started, let's start with the value iteration algorithm that we covered before.
3. It's a pretty simple algorithm and it's a little easier for us to think about, but we'll get back to the Q iteration methods a little bit later.
4. So to remind everybody, in value iteration, we can think of it as having two steps.
5. Step one, construct your table of Q values as the reward plus gamma times the expected value at the next state.
6. And then step two, set your value function to be the max over the rows of that table.
7. So you can think of it as constructing this table of values and then iterating this procedure.
8. So the question we could ask is, does this algorithm converge?
9. And if it does converge, what does it converge to?
10. So one of the ways that we can get started with this analysis is we can define an operator, which I'm going to write as script B.
11. And this operator is called the Bellman operator.
12. The Bellman operator, when applied to a value function, and remember the value function here is a table, so you can think of it as just a vector of numbers.
13. When applied to this vector of numbers, it performs the following operation.
14. First, it takes V and applies the operator.
15. And then it takes the operator T subscript A.
16. T subscript A is a matrix with dimensionality S by S, where every entry in that matrix is the probability of S prime given S comma A, where A is chosen according to that max.
17. So this is basically computing that expectation.
18. The expectation is a linear operator.
19. We multiply it by gamma and we add this vector RA.
20. The vector RA is a vector of numbers.
21. And then we have the function of rewards, where for every state, you pick the reward for the corresponding action A.
22. And then outside of this, you perform a max over A.
23. And crucially, this max is per element.
24. So for every state, we take a max.
25. So this funny way of writing the Bellman backup basically just captures the value iteration algorithm.
26. So the value iteration algorithm consists of repeatedly applying the operator B to the vector V.
27. The max comes from step 2, and the stuff inside the max comes from step 1.
28. So the reward is a stacked vector of rewards at all states for action A.
29. And T is a matrix of transitions for action A, such that T is the probability that S prime equals I, given that S equals J.
30. And we took the action A.
31. Now, one interesting property that we can show is that V star, is a fixed point of B.
32. What is V star?
33. V star is the value function for the optimal policy.
34. So if we can get V star, then we will recover the optimal policy.
35. V star is equal to the max over A of RSA plus gamma, times the expected value of V star S prime.
36. Right?
37. So if we find a value function, if we find a vector that satisfies this equation, we found the optimal value function.
38. And if we use the argmax policy with a solution, we find that the value function is equal to the maximum value function of RSA plus gamma.
39. And if we use the argmax policy with respect to that, we will get the optimal policy, the policy that maximizes total rewards.
40. So that means that V star is equal to B times V star.
41. So V star is a fixed point of B.
42. So that's very nice.
43. If we find a fixed point of B, then we'll have found the optimal value function.
44. And furthermore, it's actually possible to show that V star always exists, this fixed point always exists, it's always unique, and it always corresponds to the optimal value function.
45. So the only question that we're left with is, does repeatedly applying B to V actually find this fixed point?
46. So it's a fixed point iteration algorithm.
47. Does the fixed point iteration algorithm converge?
48. If it does converge, it will converge to the optimal policy, and it has a unique solution.
49. So will we reach it?
50. So I won't go through the proof in detail in this lecture, but the high-level sketch behind how we argue that value iteration converges, is by arguing that it's a contraction.
51. So we can prove that value iteration reaches V star because B is a contraction.
52. What does it mean to be a contraction?
53. It means that if you have any two vectors, V and V bar, then applying B to both V and V bar will bring those vectors closer together, meaning that BV minus BV bar, their norm is less than or equal to the norm of V minus V bar.
54. In fact, it's a contraction by some coefficient, and that coefficient happens to be gamma.
55. So not only is BV minus BV bar norm less than or equal to V minus V bar norm, it's actually less than or equal to V minus V bar norm times gamma.
56. So you will contract, and you'll actually contract by some non-trivial amount, which means that V and V bar will always get closer together as you apply B to them.
57. Now, the proof that B is a contraction is not actually all that complicated.
58. I just don't want to go through it on this slide, but you can look it up as a standard kind of textbook result.
59. But just to very briefly explain why showing that it's a contraction implies that value iteration converges, if you choose V star as your V bar, you know that V star is a fixed point of B.
60. So if you substitute in V star for V bar, then you get the equation BV minus V star norm is less than or equal to gamma times V minus V star.
61. Which means that each time you apply B to V, you get closer to V star.
62. So each time you change your value function by applying the non-linear operator B, you get closer to your optimum V star.
63. It's important to note here that the norm under which the operator B is a contraction is the infinity norm.
64. So the infinity norm is basically the difference for the largest operator.
65. So the infinity norm of a vector is the value of the largest entry in that vector.
66. So the state at which V and V star disagree the most, they will disagree less after you apply B.
67. So infinity norm.
68. And this is important.
69. This will come up shortly.
70. Alright, so regular value iteration can be written extremely concisely as just repeatedly applying this one step V goes to BV.
71. Now let's go to the fitted value iteration algorithm.
72. The fitted value iteration algorithm has another operation.
73. It has a step two where you actually perform the art min with respect to ϕ.
74. How can we mathematically understand that second step?
75. So the first step is basically the Bellman backup.
76. The second step trains the neural network.
77. What does a step actually do abstractly?
78. Well, one of the ways you can think of supervised learning is that you have some set of value functions that you can represent.
79. That set, if your value function is a neural network, it's actually a continuous set that consists of all possible neural nets with your particular architecture but with different weight values.
80. So we'll denote that set as a set omega.
81. In supervised learning we sometimes refer to this as the hypothesis set or the hypothesis space.
82. Supervised learning consists of finding an element in your hypothesis space that optimizes your objective.
83. And our objective is the squared difference between V5s and the target value.
84. Now what is our target value?
85. Our target value is basically BV.
86. Right, that's what we did in step one.
87. Step one is basically doing BV.
88. That's literally the equation for BV.
89. So you can think of the entire fitted value iteration algorithm as repeatedly finding a new value function, V prime, which is the art min inside the set omega of the squared difference between V prime and BV, where BV is your previous value function.
90. Now this procedure is itself actually also a contraction, right?
91. So when you perform this supervised learning, you can think of it as a projection in the L2 norm.
92. So you have your old V, you have your set of possible neural nets represented by this line.
93. So omega is basically all the points on that line.
94. The whole space is all possible value functions.
95. Omega doesn't contain all possible value functions.
96. So omega restricts us to this line.
97. When we construct BV, we might step off this line.
98. So the point BV doesn't line the set omega.
99. When we perform supervised learning, when we perform step two of fitted value iteration, what we're really doing is we're finding a point in the set omega that is as close as possible to BV.
100. And as close as possible means that it's going to be at a right angle.
101. So we'll project down onto the set omega, and it'll be a right angle projection.
102. So that'll get us V prime.
103. So we can define this as a new operator.
104. We can call this operator π for projection.
105. And we're going to say that π V is just the art min within the set omega of this objective.
106. And this objective is just the L2 norm.
107. Now π is a projection onto omega in terms of the L2 norm.
108. And π is also a contraction, because if you project something under L2 norm, it gets closer.
109. So the complete fitted value iteration can be written also in one line as just V becomes π BV.
110. So first you take a bell and back up on V, then you project it, and then you get your new V.
111. So that's our fitted value iteration algorithm.
112. B is a contraction with respect to the infinity norm, the so-called max norm.
113. So that's what we saw before.
114. π is a contraction with respect to the L2 norm, with respect to Euclidean distance.
115. So π V minus π V bar squared is less than or equal to V minus V bar squared.
116. So, so far so good.
117. Both of these operators are contractions.
118. The reason, by the way, the intuition behind why π is a contraction, is that if you have any two points in Euclidean space and you project them on a line, they can only get closer to each other, they can never get further.
119. So that's why π is a contraction.
120. Unfortunately, π times B is not actually a contraction of any kind.
121. This might at first seem surprising, because they're both contractions individually, but remember that they're contractions for different norms.
122. B is a contraction in the infinity norm, π is a contraction in the L2 norm.
123. It turns out if you put those two together, you might actually end up with something that is not a contraction under any norm.
124. And this is not just a theoretically idiosyncrasy.
125. This actually happens in practice.
126. So if you imagine that this is your starting point, the yellow star is the optimal value function, and you take a step, so your regular value iteration will gradually get closer and closer to the star.
127. If you have a projected value iteration algorithm, a fitted value iteration algorithm, then you're going to restrict your value function to this line each step of the way.
128. So your Bellman backup, Bv, will get you closer to the star in terms of infinity norm, and then your projection will move you back onto the line.
129. And while both of those operations are contractions, notice that v' is now actually further from the star than v is.
130. And you can get these situations where each step of the way actually gets you further and further from v star.
131. And this is not just a theoretically idiosyncrasy, this can actually happen in practice.
132. So the sad conclusions from all this are that value iteration does converge in the tabular case, fitted value iteration does not converge in general, and it doesn't converge in general, and it often doesn't converge in practice.
133. Now what about fitted Q iteration?
134. So far all of our talk has been about value iteration.
135. What about fitted Q iteration?
136. It's actually exactly the same thing.
137. So in fitted Q iteration, you can also define an operator B.
138. It looks a little bit different.
139. Now it's R plus gamma T times max Q, so the max is now at the target value, but same basic principle.
140. So now the max is after the transition operator.
141. That's the only difference.
142. B is still a contraction in the infinity norm.
143. You can define an operator π exactly the same way as the operator that finds the arg min in your hypothesis class that minimizes square difference.
144. You can define fitted Q iteration, as Q becomes π B Q, just like with value iteration.
145. And just like before, B is a contraction in the infinity norm, π is a contraction in the L2 norm, and π B is not a contraction of any kind.
146. This also applies to online Q learning and basically any algorithm of this sort.
147. Now at this point, some of you might be looking at this thing and thinking, something is very contradictory here.
148. We just talked about how this algorithm works, this algorithm doesn't converge, but at the core of this algorithm is something that looks suspiciously like gradient descent.
149. Like isn't this whole process just doing regression on the target values?
150. Don't we know that regression converges?
151. Isn't this just gradient descent?
152. Well, the subtlety here is that Q learning is not actually gradient descent.
153. So Q learning is not taking gradient steps on a well-defined objective.
154. It's because the target values in Q learning themselves depend on the Q values.
155. And this is also true for Q iteration.
156. But you're not considering the gradient through those target values.
157. So the gradient that you're actually using is not the true gradient of a well-defined function.
158. And that's why it might not converge.
159. Now it's probably worth mentioning that you could turn this algorithm into a gradient descent algorithm by actually computing the gradient through those target values.
160. They're non-differential because of the max, but there are some technical ways to deal with that.
161. The bigger problem is that the resulting algorithm, which is called a residual algorithm, has very, very poor numerical properties and doesn't work very well in practice.
162. In fact, even though this kind of Q learning procedure that I described is not guaranteed to converge, in practice it actually tends to work much, much better than residual gradient, which, though guaranteed to converge, has extremely poor numerical properties.
163. Okay, so short version, Q learning and Fitted Q iteration are not actually doing gradient descent, and the update is not the gradient of any well-defined function.
164. There's also, unfortunately, another sad corollary to all this, which is that our actual critic algorithm that we discussed before also is not guaranteed to converge under function approximation for the same reason.
165. So there we also do a Bellman backup when we use a bootstrap update, and we do a projection when we update our value function, and the concatenation of those is not a convergent operator.
166. So Fitted bootstrap policy evaluation also doesn't converge.
167. And by the way, one aside about terminology, most of you probably already noticed this, but when I use the term V π, I'm referring to the value function for some policy π.
168. This is what the critic does.
169. When I use V star, this is the value function for the optimal policy π star, and this is what we're trying to find in value iteration.
170. Okay, so to review, we talked about some value iteration theory, we discussed the operator for the backup, the operator for the backup, the operator for the projection.
171. This is a typo on the slide, they're not actually linear operators, but they are operators.
172. We talked about how the backup is a contraction, and how tabular value iteration converges.
173. We talked about some convergence properties with function approximation, where the projection is also a contraction, but because it's a contraction in a different norm, backup followed by projection is not actually a contraction.
174. And therefore, Fitted value iteration does not in general converge, and its implications for Q-learning are that Q-learning fitted to Q-iteration, et cetera, also do not converge when we use neural nets, when we have a projection operator.
175. This might seem somewhat somber and depressing.
176. We will find out in the next lecture that in practice, we can actually make all of these algorithms work very well, but their theoretical properties leave us with a lot to be desired.