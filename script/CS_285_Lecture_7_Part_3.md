1. All right.
2. In the next portion of today's lecture, we're going to discuss how this generic form of Fittick Q iteration that we covered can be instantiated as different kinds of practical deep reinforcement learning algorithms.
3. So first, let's talk a little bit more about what it means for Fittick Q iteration to be an off-policy algorithm.
4. So just to remind everybody, off-policy means that you do not need samples from the latest policy in order to keep running your RL algorithm.
5. Typically, what that means is that you can take many gradient steps on the same set of samples or reuse samples from previous iterations.
6. So you don't have to throw out your old samples.
7. You can keep using them, which in practice gives you more data to train on.
8. So intuitively, the main reason that Fittick Q iteration allows us to get away with using off-policy data is that the one place where the policy is actually used is actually utilizing the Q function rather than stepping through the simulator.
9. So as our policy changes, what really changes is this max.
10. Remember, the way that we got this max was by taking the argmax, which is our policy, the policy in an argmax policy, and then plugging it back into the Q value to get the actual value for the policy.
11. So inside of that max, you can kind of unpack it.
12. And pretend that it's actually Q ϕ of si prime comma argmax of Q ϕ, and that argmax is basically our policy.
13. So this is the only place where the policy shows up.
14. And conveniently enough, it shows up as an argument to the Q function, which means that as our policy changes, as our action ai prime changes, we do not need to generate new rollouts.
15. You can almost think of this as a kind of model.
16. The Q function allows you to sort of simulate what kind of values you want to get out of the model.
17. So you can simulate the values you would get if you were to take different actions.
18. And then, of course, you take the best action if you want to most improve your behavior.
19. So this max approximates the value of π prime, our greedy policy, at si prime.
20. And that's why we don't need new samples.
21. We're basically using our Q function to simulate the value of new actions.
22. So given a state and an action, the transition is actually independent of π.
23. Right?
24. If si and ai are fixed, no matter how much we change π, si prime is not going to change, because π only influences ai, and here ai is fixed.
25. So one way that you can think of Theta Q iteration kind of structurally is that you have this big bucket of different transitions, and what you'll do is you'll back up the values along each of those transitions, and each of those backups will improve your Q value.
26. But you don't actually really care so much about which specific transitions they are, so long as they kind of cover up the value of each of those transitions.
27. So you don't really care so much about which specific transitions they are, so long as they kind of cover up the value of each of those transitions.
28. So you don't really care so much about which specific transitions they are, so long as they kind of cover up the value of each of those transitions.
29. So you can imagine that you have this data set of transitions, and you're just plugging away on this data set, running Theta Q iteration, improving your Q function each time you go around the loop.
30. Now, what exactly is it that Theta Q iteration is optimizing?
31. Well, this step, the step where you take the max, improves your policy.
32. Right?
33. So in the tabular case, this would literally be your policy improvement.
34. And your step 3 is minimizing the error of fit.
35. So if you had a tabular update, you would just directly write those YIs into your table, but since you have a neural network, you have to actually perform some optimization to minimize an error against those YIs, and you might not drive the error perfectly to 0.
36. So you could think of Theta Q iteration as optimizing an error, the error being the Bellman error, the difference between Q, ϕ, SA, and those target values Y, and that is kind of the closest to an actual optimization objective.
37. But of course, that error itself doesn't really reflect the goodness of your policy.
38. It's just the accuracy with which you're able to copy your target values.
39. If the error is 0, then you know that Q, ϕ, SA is equal to RSA plus γ max A prime Q, ϕ, s' A prime.
40. And this is an optimal Q function, corresponding to the error of fit.
41. And this is an optimal Q function, corresponding to the optimal policy π prime, where the policy is recovered by the argmax rule.
42. So this you can show maximizes reward.
43. But if the error is not 0, then you can't really say much about the performance of this policy.
44. So what we know about Theta Q iteration is, in the tabular case, your error will be 0, which means that you'll recover Q star.
45. If your error is not 0, then most guarantees are lost when we leave the tabular case.
46. So that's how the error is resolved.
47. All right.
48. Now let's discuss a few special cases of Theta Q iteration, which correspond to very popular algorithms in the literature.
49. So, so far, the generic form of Theta Q learning that we talked about has these three steps.
50. Collect the data set, evaluate your target values, train your neural network parameters to fit those target values, and then alternate these two steps k times.
51. And then after k times, go out and collect more data.
52. You can instantiate a special case of this, of this algorithm with particular choices for those hyperparameters, which actually corresponds to an online algorithm.
53. So in the online algorithm, in step one, you take exactly one action, ai, and observe one transition, si, ai, si prime, ri.
54. Then in step two, you compute one target value for that transition that you just took.
55. Very much analogous to how you calculate the advantage value in actor critic, in online actor critic, for the one transition that you just took.
56. And then in step three, you take one gradient descent step on the error between your q values and the target value that you just computed.
57. So the equation that I have here, it looks a little complicated, but I basically just applied the chain rule of probability to that objective inside the art min in step three.
58. So applying chain rule, you get dq d ϕ at si, ai times the error q ϕ si, ai minus yi.
59. And the error in those parentheses, that q si, ai minus yi, is something that you can do in the algorithm.
60. So you can do that in the algorithm.
61. So you can do that in the algorithm.
62. So you can do that in the algorithm.
63. So you can do that in the algorithm.
64. So you can do that in the algorithm.
65. So you can do that in the algorithm.
66. And if you look at the iron heaps on thevil HP code sh�.
67. You can put that in you�� Ди-авこんにちは.
68. Could be World War Two or T 만� 쓸 259 World War Twoof 2014, es л ápинияzzor of the modern world, Where you can have O cereal, there's a secret combination part of these functions, center function in the X axis is a fundamental Magnus Reifz And a string in it, Can you use poss di Merl beacons,原因 of the amnesty between O원, X and V the origin of f,unku, are nint cña of the origin of F C uq0re Stiru, as mau the typical gel daymm , nth точl n Potato assy mo-vizcky, And it is an on-policy algorithm, so you do not have to take the action AI using your latest greedy policy.
69. So what policy should you use?
70. So your final policy will be the greedy policy.
71. If Q-learning converges and has error zero, then we know that the greedy policy is the optimal policy.
72. But while learning is progressing, using the greedy policy may not be such a good idea.
73. Here's a question for you to think about.
74. Why might we not want to use the greedy policy, the argmax policy, in step one while running online Q-learning or online Q-duration?
75. Take a moment to think about this question.
76. So part of why we might not want to do this is that this argmax policy is deterministic.
77. And if our initial Q function is quite bad, it's not going to be random, but it's going to be arbitrary.
78. So if we run the Q-duration algorithm, then it will essentially commit our argmax policy to take the same action every time it enters a particular state.
79. And if that action is not a very good action, we might be stuck taking that bad action essentially in perpetuity, and we might never discover that better actions exist.
80. So in practice, when we run fitted Q-duration or Q-learning algorithms, it's highly desirable to modify the policy that we use in step one to not just be the argmax policy, but to inject some additional randomness to produce better results.
81. So a很多 validate Brotherhood theory where this Eso, with probability one minus Epsilon you will take the greedy action And then with probability utility that will be probably probability epsilon, you will take one of the other actions uniformly at random.
82. So the probability of every action is 1 minus epsilon if it's the arc max, and then epsilon divided by the number of actions minus 1 otherwise.
83. This is called epsilon reading.
84. Why might this be a good idea?
85. Well, if we choose epsilon to be some small number, it means that most of the time we take the action that we think is best.
86. And that's usually a good idea, because if we've got it right, then we'll go to some good region and collect some good data.
87. But we always have a small but non-zero probability of taking some other action, which will ensure that if our Q function is bad, eventually we'll just randomly do something better.
88. It's a very simple exploration rule, and it's very commonly used in practice.
89. A very common practical choice is to actually vary the value of epsilon over the course of training.
90. And that makes a lot of sense, because you expect your Q function to be pretty bad initially, and at that point you might want to use a larger epsilon, and then as learning progresses, your Q function gets better, and then you can reduce epsilon.
91. Another exploration rule that you could use is to select your actions in proportion to some positive transformation of your Q values.
92. And a particularly popular positive transformation is exponentiation.
93. So if you take actions in proportion to the exponential of your Q values, what will happen is that the best actions will be the most frequent.
94. Actions that are almost as good as the best action will also be taken quite frequently, because they'll have similarly high probabilities.
95. But if some action has an extremely low Q value, then it will almost never be taken.
96. In some cases, this kind of exploration rule can be preferred over epsilon greedy, because with epsilon greedy, the action that happens to be the max gets much higher probability, and if there are two actions that are about equally good, the second best one has a much lower probability.
97. Whereas if there are two actions that are about equally good, the second best one has a much lower probability.
98. Whereas if there are two actions that are about equally good, the second best one has a much lower probability.
99. Whereas if there are two actions that are about equally good, the second best one has a much lower probability.
100. Whereas with this exponentiation rule, if you really have two equally good actions, you'll take them about an equal number of times.
101. The second reason it might be better is if you have a really bad action, and you've already learned that it's just a really bad action, you probably don't want to waste your time exploring it.
102. Whereas epsilon greedy won't make use of that.
103. So this is sometimes also called the Boltzmann exploration rule, also the softmax exploration rule.
104. We'll discuss more sophisticated ways to do exploration in much more detail in another lecture in the summer.
105. So until then, thanks for watching, and I'll see you in the next one.
106. the second half of the course, but these simple rules are hopefully going to be enough to implement basic versions of Q iteration and Q learning algorithms.
107. All right, so to review what we've covered so far, we've discussed value-based methods, which don't learn a policy explicitly, but just learn a value function or Q function.
108. We've discussed how if you have a value function, you can recover a policy by using the arcmax, and how we can devise this fitted Q iteration method, which does not require knowing the transition dynamics, so it's a true model-free method.
109. And we can instantiate it in various ways, as a batch mode off policy method, or an online Q learning method, depending on the choice of those hyperparameters, the number of steps we take to gather data, the number of gradient updates, and so on.