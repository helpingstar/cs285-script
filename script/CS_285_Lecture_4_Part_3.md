1. All right, in the next portion of this lecture, I'm going to introduce the notion of value functions, which are a very useful mathematical object, both for designing reinforcement learning algorithms and for conceptually thinking about the reinforcement learning objective.
2. So, as I mentioned earlier, the reinforcement learning objective can be defined as an expectation.
3. It's an expectation of a sum of rewards with respect to the trajectory distribution, or equivalently, a sum over time of the expected reward for every state action marginal.
4. Now, one of the things we could do with this expectation is we can actually write it out recursively.
5. So, you know how we can apply the chain rule of probability to factorize the trajectory distribution as a product of many distributions.
6. In the same way, we can apply the chain rule and write out an expected value with respect to that distribution as a series of nested expectations.
7. So, the outermost expectation here would be over p(s_1).
8. Inside of it, we have an expected value with respect to a_1, distributed according to π(a_1|s_1).
9.  And now, since we have an expectation for both s_1 and a_1, we can put in the first reward, r(s_1,a_1).
10. And notice that this inner expectation, the one over a_1, is conditional on s_1.
11. I have a bunch of blank space here because I'm going to need to put in all the other rewards.
12. But we already have r(s_1,a_1).
13. Now, we add to that all the other rewards, but those require putting in another expectation now over s_2, distributed according to p(s_2|s_1,a_1).
14. So, this expectation is conditioned on s_1 and a_1.
15. And inside of that, we have another expectation over a_2, distributed according to π(a_2|s_2).
16. And now, since we have both s_2 and a_2, we can put in r(s_2,a_2).
17. And then we add to that the expected value over s_3, inside of which is the expected value over a_3.
18. Inside of it is r(s_3,a_3), and so on and so on.
19. And we have these nested expectations.
20. Now, at first, it kind of seems like we just wrote a very concise expected value over trajectories as a really, really messy set of nested expectations.
21. But one thing that we could think about is, well, what if we had some function that told us the stuff that goes inside of the second expectation?
22. What if we had some function that told us r(s_1,a_1) plus the expected value over s_2 plus et cetera, et cetera, et cetera?
23. So, what if we knew this part?
24. So, let's define a symbol for this.
25. Let's say that Q(s_1,a_1) is equal to r(s_1,a_1) plus the expectation over s_2 of the expectation over a_2 of r(s_2,a_2), et cetera.
26. So, basically, just this middle part, the part that goes inside the second set of square brackets, I'm just going to call that Q(s_1,a_1).
27. Then we can write our original RL objective as simply the expected value over s_1 of the expected value over a_1 of Q(s_1,a_1).
28. So, it's just a little bit of symbolic manipulation, a little bit of definition.
29. But the important point about this definition is that if you knew Q(s_1,a_1), then optimizing the policy at the first time step would be very easy.
30. So, if you had access to Q(s_1,a_1) and you needed to select the policy π(a_1|s_1), you would just select the policy for which this expected value is largest.
31. You could simply test every action and just assign 100% probability to the best one, one with the largest value for Q.
32. So, this basic idea can be extended to a more general concept.
33. So, this is the simple rule that I said, you know, a simple way to get π here is just assign a probability of 1 to the argmax.
34. So, the more general principle is what we're going to call the Q function.
35. So, the Q function can be defined at other time steps, not just time step 1.
36. And the definition is this.
37. Q^π(s_t,a_t).
38. And I say Q^π because it depends on π.
39. Q^π(s_t,a_t) is equal to the sum over all time steps from T until the end, capital T, of the expected value of the reward at that future time step, conditioned on starting in (s_t,a_t).
40. So, what that means is basically if you start in (s_2,a_2) and then roll out your policy, for the rest of time will be the expected sum of rewards.
41. A closely related quantity that we can also define is something called the value function.
42. The value function is defined in much the same way, only it's conditioned on only a state, rather than a state and action.
43. So, the value function says if you start in state s_t and then roll out your policy, what will be your expected total value?
44. And the value function can also be written as the expected value over actions of the Q function.
45. Right, because if the Q function tells you the expected total reward if you start in (s_t,a_t), then taking the expectation of that with respect to a_t will give you the expected total reward if you start in s_t.
46. So now, one observation we could make is the expectation of the value function at state s_1 is the entirety of the reinforcement learning objective, for the same reason that the expected value with respect to s_1, a_1 of Q(s_1,a_1) was the RL objective on the previous slide.
47. Okay, so at this point, I would like everyone to pause for a minute and think about these definitions of Q functions and value functions.
48. You might want to flip back to the previous slide if something here is unclear.
49. Take a moment to think about that.
50. And if something about these definitions is unclear, please make sure to write a question in the comments.
51. All right, let's continue.
52. So what are Q functions and value functions good for?
53. Well, I provided some intuition for this a couple slides ago when I talked about how once you have a Q function for at least the first time step, you can recover a better policy for the first time step.
54. So one idea is that if we have a policy π and we can figure out its full Q function, Q^π(s,a), then we can improve π.
55. For example, we can pick a new policy π' that assigns a probability of 1 to given action if that action is argmax of Q^π(s,a).
56. And we can do this on the first time step, but also on all of the time steps.
57. And in fact, we can show that this policy is at least as good as π and probably better.
58. Don't worry if it's not obvious to you right now why this is true.
59. We will cover this in much more detail later.
60. But this is the basis of a class of methods called policy iteration algorithms, which themselves can be used to derive Q learning algorithms.
61. And crucially, it doesn't matter what π is you can always improve it in this way.
62. Another idea which will use in the next lecture when we talk about policy gradients, is you can use this to compute a gradient to increase the probability of a good action a.
63. So the intuition is that if Q^π(s,a) is larger than V^π(s), then a is better than average.
64. Because remember that V^π(s) is just the expected value of Q^π(s,a) under π(a|s).
65. By this definition, V^π(s) is how you will do on average when you use your policy from state s.
66. So if you can do better than average, if you can choose an action a so that Q^π(s,a) is larger than V^π(s), then you will do better.
67. You'll do better than average under your old policy.
68. So one thing you could do is you could modify π(a|s) to increase the probability of actions whose value under the Q function is larger than the value at that state.
69. And you can actually use this to get a gradient-based update rule on π.
70. These ideas are very important in RL, and we'll revisit them again and again in the next few lectures when we talk about model-free reinforcement learning algorithms.
71. All right.
72. So in the anatomy of the reinforcement learning algorithm, the green box is typically where you would use or where you would learn Q functions or value functions.
73. So Q functions and value functions fundamentally are objects that evaluate how good your policy currently is.
74. So you would typically fit them or learn them in the green box, and then use them in the blue box to improve the policy.