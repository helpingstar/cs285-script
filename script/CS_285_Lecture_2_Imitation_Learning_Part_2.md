1. Part two of this lecture will be perhaps the most mathematically involved, because we're going to discuss some of the formal explanations behind why behavioral cloning doesn't in general produce good results.
2. And this formal explanation will actually help us to develop some of the solutions in the future.
3. Okay, so let's go back to this intuitive picture, which I used to argue that the reason that behavioral cloning doesn't do so well is because even if you learn a very good policy, if that policy makes a small mistake, it'll put you into a situation that's a little bit different from the ones that it was trained on, where it's more likely to make a bigger mistake, which in turn will put you into an even more unfamiliar situation.
4. And from there, the mistakes might build up and up.
5. So to try to make this more precise, let's start introducing a little bit of notation.
7. We have a policy, pi theta at given ot, and that policy is trained using a training set that comes from a particular distribution.
8. And that distribution is produced by a person providing demonstrations, like a person driving a car, for example.
9. So I'm going to use p data ot to denote the distribution that produced the training set.
10. Now, p data ot might be a very complex distribution.
11. We don't really care about that.
12. All we care about at this stage is that it is whatever distribution over observations comes from the human's drive.
13. And then I'll use a different symbol to denote the distribution over observations that the policy itself sees when it's driving a car.
14. And that's going to be denoted with p pi theta ot.
15. And of course, because the policy doesn't drive exactly the same way that a person drives, p pi theta ot is not going to be the same as p data ot.
16. So if we want to understand just how different these things are going to be, let's first discuss how pi theta is trained.
17. Well, pi theta is trained under p data ot, which means that we can, if we're using some standard training objective, like supervised maximum likelihood or empirical risk minimization, basically, we can write the objective as the following.
18. It's maximizing the log probability of the actions from the human given the observations.
21. And the observations are sampled from p data ot.
22. So the expectation is under p data ot.
23. Now, we know from supervised learning theory that if we train our policy in this way and we don't overfit and we don't underfit, then we would expect the log probability of the actions under the distribution p data ot to be high.
24. We would expect good actions to have high probability.
25. Of course, the problem is that performance of the policy is not determined by the log probability that assigns to good actions under the experts observation distribution, but under the testing distribution which is p pi theta.
26. So the log probability of good actions under p pi theta might be very different because p data and p pi theta are not the same.
27. This is often referred to as distributional shift, which means that the distribution under which the policy is tested is shifted from the distribution under which it's trained.
28. Now it just so happens that that shift is due to the policy's own mistakes, but this is the formal statement for why we can't in general expect it to be correct.
29. And it's pretty easy to construct counter examples where this will be very bad.
30. So before I construct that counter example, let me set things up a little bit more precisely, which will make the analysis more concise.
31. So first we have to define the what we want?
33. What determines whether our policy is good or bad?
34. So it's trained to maximize the likelihood of the training actions, but presumably that's not all we want.
35. We want some other notion of goodness, like it has to actually drive the car well.
36. So what makes a learned policy good or bad?
37. Well it's a choice that we make, it's a design choice.
38. It probably shouldn't be the likelihood of the training actions, because a policy could assign very high probability to the actions of the human driver took in the kinds of states that they actually saw, but then take completely incorrect actions in the states that are even a little different.
39. So we probably need a better measure of goodness that we can use to analyze our policies.
40. And one measure that we can use is we can define a cost.
41. And the cost is a function of states and actions, and we'll say that the cost is zero if the action is the same as the human driver's action.
42. So let's assume the human driver has a deterministic policy.
43. It's not hard to extend this to stochastic policies, but it makes a lot of the notation very complex.
44. So we'll just say that say that the human driver has a deterministic policy, Pi star, the cost is zero if the action matches what they would have done, and it's 1 otherwise.
46. And that's a very convenient cost to define, because you can basically say that whenever the policy that you learn makes a mistake, you pay a cost of 1.
47. So the total cost is basically the number of mistakes you're going to make.
48. Notice here that I started mixing up S and O.
49. Don't worry about that.
50. So all of the analysis here will be in terms of S.
51. It's a little bit involved to extend this to O, to extend it to partially observed settings.
52. So this is one of those cases where the mock-up property is very useful.
53. It is possible to do.
54. It'll just make everything more complicated to write.
55. So we'll kind of transparently switch from O to S for this section and not worry about it.
56. I warned you that I would do that.
57. Okay, so our goal now is going to be to minimize the expected cost, meaning the expected number of mistakes that our policy is going to make, but expect that under what distribution?
58. Well, what we care about is the number of mistakes that the policy makes when it actually drives the car.
59. We don't really care how many mistakes it would make when it's looking at the human's images, because that's not how it's going to be used.
60. So what we care about is the cost in expectation under p pi theta, under the distribution of states the policy will actually see.
61. And that's a very, very important distinction, because we're training the policy to assign high probability to the actions under p data.
62. But what we really care about is to minimize the number of mistakes under p pi theta.
63. Okay?
64. So that's an important distinction.
65. So in analyzing how good behavioral cloning is, what we're really trying to do is we're trying to say, well, if we succeeded in doing our supervised learning well, what can we say about this expected value of this cost under p pi theta?
66. So basically, will we successfully minimize the number of mistakes when we run the policy?
67. Yes or no?
68. Okay.
69. So let's work on that problem a little bit.
70. So here's our picture.
71. Our total horizon length is capital T.
72. So that's how long each of these trajectories are.
73. This is our cost.
74. And we're going to make some assumption, which basically amounts to saying, let's assume that supervised learning worked.
75. So our assumption, the simple one that we'll start with is we'll just say that the probability assigned to any action that is not the expert's action is less than or equal to epsilon if the state S is one of the training states.
76. So this basically says that on the training states, your probability of making a mistake is small.
77. It's going to be some small number epsilon.
78. In general, we can extend this to say the probability is small for any state that's sampled from the training distribution.
79. So it doesn't have to literally be one of the states that you saw.
80. But for now, just to keep it simple, let's say that it's literally one of the states that you saw.
81. And now let's construct a very simple problem.
82. Where under this assumption, if you assume that your probability of making a mistake is epsilon for any state that you saw, and unbounded for any state that you didn't see, things are going to be very bad.
83. And I'm going to call this the tightrope walker example.
84. So imagine that you have a problem where at every state, there's a very specific good action, which is to stay on the tightrope.
85. And if you make an incorrect action, if you make a mistake, then you fall off the tightrope.
86. Now, falling off the tightrope is not actually bad in the sense that you hurt yourself.
87. Let's say that there's a safety net or something.
88. It's bad because you'll find yourself in a state that the expert never saw.
89. So the expert that was providing with demonstrations never fell off the tightrope.
90. The bad thing about falling off is that you are in an unfamiliar place.
91. So think of it as a discrete environment with, on a grid.
92. So the gray squares represent the squares that are on the tightrope.
93. The red ones are where you fall off.
94. The demonstrations always go steadily to the right.
95. So the action is always to go to the right.
96. If you make a single mistake, if you go up or down, then you fall off the tightrope.
97. And you're very concerned about that, not because you'll hurt yourself, but because you won't know what to do in that situation afterwards.
98. So how many mistakes will you make over the course of a trajectory on average if your probability of making a mistake is less than or equal to epsilon at every state on the tightrope?
99. So what we want to do is we want to write down a bound on the total cost.
100. So on the first time step, your probability of making a mistake is epsilon.
101. If you make a mistake, you fall off the tightrope, all of the remaining time steps are also in general going to be mistakes because you have no idea what to do.
102. So for the first time step, you incur at least epsilon times capital T mistakes on average.
103. Now with probability one minus epsilon, you didn't make a mistake.
104. So then you move on to the next time step, the second square.
105. And in the second square, you again have an epsilon probability of making a mistake, in which case, you fall off the tightrope.
106. And the remaining T minus one time steps are spent flailing around off the tightrope and making capital T minus one mistakes because that's how many time steps are left.
107. And then with probability one minus epsilon, you go on to the third step and so on and so on.
108. So you have this series where you add up all these terms.
109. There are capital T terms.
110. And each of those capital T terms is on the order of epsilon T.
111. Because if you assume that epsilon is a small number, one minus epsilon is negligibly small.
112. So the order of all these terms, one minus epsilon is negligibly close to one.
113. So the order of all these terms is going to be about epsilon times capital T.
114. And there is capital T of those terms.
115. So that means that the number of mistakes on the order of epsilon capital T squared.
116. It's like epsilon capital T squared over two with some correction term for one minus epsilon.
117. But this is basically the order of that for small values of epsilon.
118. Now, what does this tell us about behavioral cloning?
119. Well, it tells us that it's actually very bad because if you make a very long tightrope, this quadratic increase in the number of mistakes will really get you in trouble.
120. What we would really like is a linear increase in the number of mistakes.
121. So it's reasonable that the longer you go, the more mistakes you're going to accumulate.
122. But if the rate is more than linear, then long horizons are getting us into a lot of trouble.
123. Okay, so we're getting epsilon T squared.
124. Now, this is a counterexample.
125. This shows that in the worst case, you will get epsilon T squared.
126. It turns out that in general, epsilon capital T squared is actually the bound.
127. So you won't do worse than epsilon T squared.
128. That's not actually necessary to understand that behavioral cloning is bad, but we can actually bound how bad it is.
129. It's just not a very good bound.
130. And we're actually going to derive that because the kind of analysis that we'll use for that can be pretty useful in all sorts of other topics in reinforcement learning.
131. So I like to go through it.
132. Just to give a sense for how these dynamical systems can be analyzed.
133. Okay, so we showed that in the worst case, you get epsilon T squared.
134. Next, we'll show that you won't do worse than epsilon T squared, meaning that there's a bound of epsilon T squared in general.
135. Okay, so here we will actually have a more general analysis.
136. So instead of saying that all of your states literally come from your training set, we'll say our states are sampled from P train.
137. So for any state sampled from P train, uh, your error is less than or equal to epsilon.
138. It's actually enough to just assume that the expected value of the error is less than or equal to epsilon, which is more realistic, of course, because typically you train for the expected value of the, uh, of the loss.
139. And with DAG, with, uh, with DAG, which we'll talk about later, it's an algorithm that we'll introduce at the end of the lecture.
140. It'll make this problem go away because it'll make P train and P by theta the same, but for now they're not the same.
141. So that's going to be a problem.
142. We're going to show that the expected number of mistakes is, uh, going to be epsilon T squared in the worst case.
143. And then of course with DAG, or when they become equal, it will be epsilon T.
144. Uh, so that'll the DAG stuff, don't worry about it yet.
145. That'll come at the end of the lecture.
146. Uh, but if P train is not equal to P theta, uh, then here's what happens.
147. What we can do if we want to figure out the expected value of the cost, uh, is we can describe the distribution over states at time step T as a sum of two terms.
148. And one of those terms is going to be easy to analyze and the other term will be really complicated and we'll just use a bound for that.
149. So we can say that at time step T there's some probability that you didn't make any mistakes at all, meaning some probability that you stayed on the tightrope, that you did everything right.
150. And if the probability of making a mistake at each step step is epsilon and you start off at an indistribution state, uh, meaning you start off at a state sample from P train, then the probability that you made no mistakes for T time steps is just one minus epsilon to the power T, little t.
151. So we can say that P theta ST is equal to one minus epsilon to the power T times P train ST because that's the probability that you didn't make a mistake.
152. And if you didn't make a mistake that you're still in the distribution P train plus one minus that one minus one minus epsilon to the T times some other distribution.
153. So it's just saying that, there's some part of your distribution for all the possibilities where you didn't do anything wrong and then there's everything else.
154. And the weight on the part where you did nothing wrong is one minus epsilon to the T and the distribution there is P train.
155. Okay?
156. So this is a decomposition you can make.
157. Now P mistake is something really complicated, right?
158. So we don't really understand what P mistake is.
159. It's like the part of P theta that is separate from P train.
160. So we're not going to make any assumptions on P mistake other than that it only constitute a one minus one minus epsilon to the T portion of your distribution, okay?
161. and of course if epsilon is very very small then the sum is dominated by the first term so if epsilon is very small then 1 minus epsilon is almost 1 so most of it is in p train but of course the larger t is the more that exponent is going to hurt okay so that's what we've got and now what i'm going to do is i'm going to relate the distribution p theta st to the distribution p train st now when you see me using this absolute value sign what i'm in general going to be referring to is a total variation divergence a total variation divergence is just a sum over all of the states of the absolute value of the difference in their probabilities it can be viewed as a very simple notion of divergence between distributions but for now we're just going to do this at one state so at any given state the absolute value of p theta st minus p train st
161. Well it's pretty to work out if you just substitute in the equation above for p theta st you'll see there's a p train term that cancels out so you get a 1 minus epsilon to the t p train minus p train so that you can take out as a 1 minus 1 minus epsilon to the t and now you end up with this equation you get it up with 1 minus 1 minus epsilon to the t times the absolute value of p mistake minus p train okay now that's still a kind of a cryptic equation we don't know what this absolute value is but we know that all probability all probabilities have to be between 0 and 1.
162. so the biggest difference between two probabilities can be at most 1 right because the worst case is p mistake is 1 and p train is 0 or vice versa and the largest total variation divergence meaning if you sum over all of the states and you try to evaluate the absolute value of their difference is going to be 2 because the worst case is that in one state one of the probabilities one the other is 0 and in some other state it's the other way around one of them is 0 is the other one so the worst possible difference between two distributions when you sum over all the states is 2.
163. so that means that this whole thing is bounded by 2 times 1 minus 1 minus epsilon to the t so what we've shown is that the total variation divergence between p theta st and p train st is 2 times 1 minus 1 minus epsilon to the t and these exponents are a little hard to deal with but for values of epsilon between 0 and 1 there's a very convenient identity that 1 minus epsilon to the t is greater than or equal to 1 minus epsilon times t so that's true for any epsilon between 0 and 1 is just an algebraic identity so if we substitute that into the inequality above we can further bound this by 2 times epsilon t and that's just an algebraic convenience so that we can get rid of these exponents it gives us a slightly looser bound but it's a little easier to think about so what we've shown now is that the total variation divergence between p theta st and p train st is bounded by 2 times epsilon t and remember total variation divergence is just the sum over all of the s's of the absolute value of the difference of their probabilities
163. Okay, so now let's talk about the quantity that we actually care about what is uh what kind of bound can we derive based on this for the sum over all of the time steps of the expected value of our cost.
163. Well, uh to figure that out we'll substitute in the equation for an expected value so an expected value is just a sum over all the states of the probability of that state times its cost and what i'm going to do is i'm going to replace p theta with p theta minus p train plus p train so I can subtract p train.
163. I can add p train in both cases that's totally fine to do.
163. And then I'll put an absolute value symbol around the p theta minus b train part because if you have some quantity you take its absolute value you can only make it bigger because if it was positive it stays where it is and if it was negative it becomes a larger value okay and that gets us this bound so in this bound what i've done is i've replaced the sum over st of the absolute value of p theta st minus p train st times ct with the total variation divergence times c max the largest value of the cost in any state so i have one portion which is p train times the cost and then i have another portion which is the total variation divergence times the maximum cost in any state so just to repeat how this step was produced first you replace p theta st with p theta st plus p train minus p train the plus p train term becomes that first term in the bound and and then I'm left with p theta minus p train.
164. I can take the absolute value of that.
165. I can sum it over all the states, and that gives me the total variation divergence.
166. And to account for the fact that in every state I have a different cost, I'll just replace that cost with the maximum cost, which I can take outside of the summation.
167. And that gives me a valid upper bound.
168. Now, at this point, I'm going to use my bound for that total variation divergence for the difference between p theta and p train, and that's 2 epsilon t.
169. And, of course, I know that my cost in p train, my expected cost, is epsilon because that's my initial assumption.
170. So the first term becomes, the first term sum over st of p train times c is epsilon.
171. The second term is 2 epsilon t times c max, and c max, of course, is 1.
172. So the largest cost I can get in any state is 1 because I can make at least one mistake in any state.
173. So that's my bound.
174. Now, notice that this is summed over t.
175. So I have t terms that are each on the order of epsilon t.
176. So that means that this is going to have a linear term and a quadratic term, which means that the overall order is epsilon t squared.
177. So what have I actually shown?
178. I've shown on the previous slide that in the worst case, you're going to get epsilon t squared, and that's the tightrope walk, for example.
179. I've also shown that epsilon t squared is, in fact, a bound, meaning that you will not do worse than epsilon t squared.
180. So that's the behavioral cloning result.
181. The behavioral cloning is epsilon t squared in the worst case, and epsilon t squared is, in fact, the bound for behavioral cloning.
182. Okay, so that's the analysis, and it's good to understand this analysis.
183. We'll talk about it more in class.
184. But the next point I want to make about this is that this is rather pessimistic.
185. And of course, we saw in the driving videos before that in practice, behavioral cloning can work.
186. So why is this rather pessimistic?
187. Well, the pessimism can be seen in the tightrope walker example.
188. The tightrope walker example is a little bit pathological in the sense that, although it is a valid decision-making problem, the fact that even a single mistake immediately puts you into an unrecoverable situation is actually quite bad.
190. So in reality, we can often recover from mistakes.
191. But the trouble is that that doesn't necessarily mean that imitation learning will always allow us to do that.
192. So a lot of the methods that make naive behavioral cloning work basically try to leverage the fact that you can recover from mistakes and somehow modify the problem to make it easier for imitation learning to learn how to do that.
193. So why, for example, does that left-right camera trick work?
194. Well, maybe the left-right camera trick is really teaching the policy how to recover from mistakes by showing it what happens when it sees an image to the left and what it should do there, and by telling it what it should do when it sees an image to the right.
195. It tells it that not only can you recover from mistakes, but here is the action that is suitable for doing that.
196. And in general, you could imagine that with these accumulating errors, if instead of training on fairly narrow, very optimal trajectories, if you instead have many trajectories that all make some mistakes and then recover from those mistakes, such that the training distribution is a little bit broader, so that whenever you make a small mistake, you're still in distribution, then your policy might actually learn to correct those mistakes and still do fairly well.
197. And that is actually one of the ideas that people tend to use somewhat heuristically to make behavioral cloning work in practice.
198. So the paradox here is that imitation learning actually works better if the data has more mistakes and therefore more recoveries.
199. So higher quality, more perfect data can actually make imitation learning work worse.
200. So that's what we'll talk about next.
