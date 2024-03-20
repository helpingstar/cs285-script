1.  All right, in the next portion of this lecture, I'm going to introduce the notion of value functions, which are a very useful mathematical object, both for designing reinforcement learning algorithms and for conceptually thinking about the reinforcement learning objective.
2.  So, as I mentioned earlier, the reinforcement learning objective can be defined as an expectation.
3. It's an expectation of a sum of rewards with respect to the trajectory distribution, or equivalently, a sum over time of the expected reward for every state action marginal.
4. Now, one of the things we could do with this expectation is we can actually write it out recursively.
5. So, you know how we can apply the chain rule of probability to factorize the trajectory distribution as a product of many distributions.
6. In the same way, we can...
7. apply the chain rule and write out an expected value with respect to that distribution as a series of nested expectations.
8. So, the outermost expectation here would be over P of S1.
9. Inside of it, we have an expected value with respect to A1, distributed according to pi of A1 given S1.
10. And now, since we have an expectation for both S1 and A1, we can put in the first reward, R of S1 comma A1.
11. And notice that this inner expectation, the one over A1, is conditional on S1.
12. I have a bunch of blank space here because I'm going to need to put in all the other rewards.
13. But we already have R of S1 A1.
14. Now, we add to that all the other rewards, but those require putting in another expectation now over S2, distributed according to P of S2 given S1 A1.
15. So, this expectation is conditioned on S1 and A1.
16. And inside of that, we have another expectation over A2, distributed according to pi of A2 given S2.
17. And now, since we have both S2 and A2, we can put in R of S2.
18. And then we add to that the expected value over S3, inside of which is the expected value over A3.
19. Inside of it is R of S3 A3, and so on and so on.
20. And we have these nested expectations.
21. Now, at first, it kind of seems like we just wrote a very concise expected value over trajectories as a really, really messy set of nested expectations.
22. But one thing that we could think about is, well, what if we had some function that told us the stuff that goes inside of the second expectation?
23. What if we had some function that told us R of S1 comma A1 plus the expected value over S2 plus et cetera, et cetera, et cetera?
24. So, what if we knew this part?
25. So, let's define a symbol for this.
26. Let's say that Q of S1 comma A1 is equal to R of S1 comma A1 plus the expectation over S2 of the expectation over A2 of R of S2 A2, et cetera.
27. So, basically, just this.
28. This middle part, the part that goes inside the second set of square brackets, I'm just going to call that Q of S1 comma A1.
29. Then we can write our original RL objective as simply the expected value over S1 of the expected value over A1 of Q of S1 comma A1.
30. So, it's just a little bit of symbolic manipulation, a little bit of definition.
31. But the important point about this definition is that if you knew Q, then optimizing the policy at the first time step would be very easy.
32. So, if you had access to Q of S1 comma A1 and you needed to select the policy pi of A1 given S1, you would just select the policy for which this expected value is largest.
33. You could simply test every action and just assign 100% probability to the best one, one with the largest value for Q.
34. So, this basic idea is that Q of S1 comma A1 is equal to R of S2 of S2 of S1 comma A1.
35. This basic idea can be extended to a more general concept.
36. So, this is the simple rule that I said, you know, a simple way to get pi here is just assign a probability of 1 to the arc max.
37. So, the more general principle is what we're going to call the Q function.
38. So, the Q function can be defined at other time steps, not just time step 1.
39. And the definition is this.
40. Q pi of S2 comma A2.
41. And I say Q pi because it depends on pi.
42. Q pi of S2 comma A2 is equal to the sum over all time steps from T until the end, capital T, of the expected value of the reward at that future time step, conditioned on starting in S2 comma A2.
43. So, what that means is basically if you start in S2 comma A2 and then roll out your policy, for the rest of time will be the expected sum of rewards.
44. A closely related quantity that we can also define is something called the value function.
45. The value function is defined in much the same way, only it's conditioned on only a state, rather than a state and action.
46. So, the value function says if you start in state S, T and then roll out your policy, what will be your expected total value?
47. And the value function can also be written as the expected value over actions of the Q function.
48. Right, because if the Q function tells you the expected total reward if you start in S, T comma A, T, then taking the expectation of that with respect to A, T will give you the expected total reward if you start in S, T.
49. So now, one observation we could make is the expectation of the value function at state S1 is the entirety of the reinforcement learning objective, for the same reason that the expected value with respect to S1, A1 of Q S1, A1 was the RL objective on the previous slide.
50. Okay, so at this point, I would like everyone to pause for a minute and think about these definitions of Q functions and value functions.
51. You might want to flip back to the previous slide if something here is unclear.
52. Take a moment to think about that.
53. And if something about these definitions is unclear, please make sure to write a question in the comments.
54. All right, let's continue.
55. So what are Q functions and value functions good for?
56. Well, I provided some intuition for this a couple slides ago when I talked about how once you have a Q function for at least the first time step, you can recover a better policy for the first time step.
57. So one idea is that if we have a policy pi and we can figure out its full Q function, Q pi S comma A, then we can improve pi.
58. For example, we can pick a new policy pi prime that assigns a probability of 1 to the value of S1.
59. And we can do this on the first time step, but also on all of the time steps.
60. And in fact, we can show that this policy is at least as good as pi and probably better.
61. Don't worry if it's not obvious to you right now why this is true.
62. We will cover this in much more detail later.
63. But this is the basis of a class of methods called policy iteration algorithms, which themselves can be used to derive Q learning algorithms.
64. And crucially, it doesn't matter what policy you use.
65. You can use it to derive the same thing.
66. So let's look at the following.
67. So let's say that you have a policy pi prime that is good for pi.
68. And you can do this on the first time step.
69. And you can do this on the second time step.
70. And you can do this on the third time step.
71. And you can do this on the fourth time step.
72. So let's say we have a policy pi.
73. And the policy has a gradient as high as pi.
74. And we can do this on all of the time steps.
75. But this is the basis of a class of methods called policy iteration algorithms, which themselves can be used to derive Q learning algorithms.
76. And crucially, it doesn't matter what pi is.
77. You can always improve it in this way.
78. Another idea, which we will use in the next lecture when we talk about policy gradients, is you can use this to compute a gradient to increase the probability of a good action in the network.
79. That's a very good example.
80. Let's do this.
81. a.
82. So the intuition is that if q pi s a is larger than v of s, then a is better than average.
83. Because remember that v pi of s is just the expected value of q pi s a under pi of a given s.
84. By this definition, v pi of s is how you will do on average when you use your policy from state s.
85. So if you can do better than average, if you can choose an action a so that q pi s a is larger than v pi of s, then you will do better.
86. You'll do better than average under your old policy.
87. So one thing you could do is you could modify pi of a given s to increase the probability of actions whose value under the q function is larger than the value at that state.
88. And you can actually use this to get a gradient-based update rule on pi.
89. These ideas are very important in RL, and we'll revisit them in a few minutes.
90. And we'll talk about them again and again in the next few lectures when we talk about model-free reinforcement learning algorithms.
91. All right.
92. So in the anatomy of the reinforcement learning algorithm, the green box is typically where you would use or where you would learn q functions or value functions.
93. So q functions and value functions fundamentally are objects that evaluate how good your policy currently is.
94. So you would typically fit them or learn them in the green box, and then use them in the blue box to improve the policy.