1. So why do we have so many different RL algorithms?
2. Why is it that we can't just teach you one RL algorithm in a couple lectures and be done with it?
3. Why do we need an entire course?
4. Well, these RL algorithms have a number of trade-offs that will determine which one works best for you in your particular situation.
5. So one important trade-off between different algorithms, and we'll touch on this as we go through the next few lectures, is sample efficiency.
6. Meaning when you execute the stuff in this orange box, when you generate samples in the environment, how many samples will you need before you can get a good policy?
7. Another trade-off is stability and ease of use.
8. Reinforcement learning algorithms can be quite complex.
9. They require trading off a number of different parameters, how you collect samples, how you explore, how you fit your model, how you fit your value function, how you update your policy.
10. Each of these trade-offs and each of these choices often introduce additional hyperparameters, which can sometimes be difficult to select for your particular problem.
11. Different methods will also have different assumptions.
12. For example, do they handle stochastic environments or can they only handle deterministic environments?
13. Do they handle continuous states and actions?
14. Can they only handle discrete actions or can they only handle discrete states?
15. Do they handle episodic problems, meaning problems with a fixed capital T horizon, or do they handle infinite horizon problems, where T goes to infinity, or both?
16. Different things are easy or hard in different settings.
17. For example, in some settings, it might be easier to represent a policy, even if the physics of the environment are very, very complex, while in other settings, it might be easier to learn a model than it is to learn the policy directly.
18. Each of these trade-offs will involve making some set of design choices.
19. For instance, you might opt for an algorithm that is not very sample efficient for the sake of having something that is easier to use, or maybe for the sake of having something that can handle stochastic and partially observed problems.
20. Or you might opt for a very efficient algorithm because your samples are very expensive, but then be willing to accommodate some other limitations like, for example, only allowing for discrete actions.
21. So typically we have to make these trade-offs depending on the particular problem that we're facing.
22. Let's talk about sample efficiency first because it's a pretty big one.
23. So sample efficiency refers to how many samples we need to obtain a good policy.
24. Basically, how many times do we have to sample from our policy until we can make it perform well.
25. That's the orange box.
26. One of the most important questions in determining the sample efficiency of an algorithm is whether the algorithm is what's called an off-policy algorithm or not.
27. An off-policy algorithm is an algorithm that can improve the policy by using previously collected samples and on-policy algorithm has to throw out all of its samples each time the policy changes even a little bit and generate new samples.
28. For this reason on-policy algorithms can be a lot less efficient.
29. For instance, a policy gradient algorithm which is a on-policy algorithm must collect new samples each time it takes a gradient step on the policy because each time the policy changes even a little bit new samples must be collected.
30. So in general, if we want to look at a kind of spectrum of with more efficient algorithms on the left and less efficient algorithms on the right, a major dividing line on the spectrum is whether it's an on-policy or an off-policy algorithm, where on the extreme end of less efficient algorithms will be things like evolutionary or gradient-free methods, then on-policy policy gradient algorithms, then Actor-Critic style methods, which can be either on-policy or off-policy, then purely off-policy methods like Q-learning, then maybe model-based deep RL methods, model-based shallow RL methods, and so on.
31. But then we could say, well, why would we ever want to use a less efficient algorithm?
32. So it seems like we should just go with the stuff on the left end of the spectrum.
33. Well, it's because the other trade-offs might not be in our favor as we move to the left.
34. For example, wall clock time, the amount of computation the algorithm needs, is not the same as sample efficiency.
35. So maybe generating samples for your application is actually very cheap.
36. Maybe you're using a very, very fast simulator.
37. For example, if you're learning how to play a game like chess, simulating chess is very, very fast.
38. So most of your computation time will go into updating your value functions, models, and policies.
39. In that case, you probably don't care nearly as much about sample efficiency.
40. And interestingly enough, the wall clock time for these algorithms is often flipped.
41. So if your simulation is very cheap, you might actually find the stuff on the right end of the spectrum to be computationally less expensive, and the stuff on the left side of the spectrum to be computationally much more expensive.
42. Stability and ease of use.
43. When it comes to stability and ease of use, we might ask questions like, does our algorithm converge?
44. Meaning if we run it long enough, is it guaranteed to eventually converge to a fixed solution, or will it keep oscillating or diverging?
45. And if it does converge, what does it converge to?
46. Does it converge to a local optimum of the RL objective?
47. Or a local optimum of any other well-defined objective?
48. And does it converge every time?
49. Coming from an optimization or supervised learning background, you might wonder at this point, why is any of this even a question?
50. Because typically when we deal with supervised learning or kind of well-defined, especially convex optimization methods, essentially we only care about methods that converge.
51. In reinforcement learning, convergent algorithms are actually a rare luxury.
52. And many methods that we use in practice, are not guaranteed to converge in general.
53. So the reason for this is that reinforcement learning often is not pure gradient descent or pure gradient ascent.
54. Many reinforcement learning algorithms are actually fixed point algorithms that only carry convergent guarantees under very simplified tabular discrete state assumptions, which often do not hold in practice.
55. And in theory, the convergence of many of the most popular RL algorithms, such as Q-learning algorithms, is actually an open problem.
56. So Q-learning is a fixed point iteration.
57. Model-based reinforcement learning is a kind of a peculiar case, because the model is not actually optimized with respect to the RL objective.
58. The model is optimized to be an accurate model.
59. The model training itself is convergent, but there's no guarantee that getting a better model will actually result in a better reward value.
60. Policy gradient is gradient descent, or technically gradient ascent, but also the least efficient of the bunch.
61. Value function fitting is a fixed point iteration, and at best it minimizes error of fit.
62. It minimizes what's called Bellman error, meaning is your value function predicting values accurately?
63. But that's not the same as saying, does your value function produce a policy with good rewards?
64. And at worst, value function fitting doesn't even minimize the Bellman error.
65. At worst, it actually might even diverge.
66. Many popular deep RL value fitting algorithms are not guaranteed to converge to anything in the nonlinear case, in the case where you use neural networks.
67. Model-based RL, and the model minimizes error of fit, which will definitely converge to a good model, but there's no guarantee that a good model will lead to a better policy.
68. Policy gradient is the only one that actually performs gradient ascent on the true objective, but as I said, it's the least efficient of the bunch.
69. Assumptions.
70. One common assumption that many RL algorithms will make is full observability, meaning that you have access to states rather than observations, or, put another way, the thing that you're observing satisfies the Markov property.
71. So, no cars driving in front of cheetahs.
72. This is generally assumed by most value function fitting methods.
73. It can be mitigated by adding things like recurrence and memory, but in general can be a challenge.
74. Another common assumption, this one is common with policy gradient methods, is episodic learning.
75. So here's a robot performing episodic learning.
76. You can see that it makes a trial, then resets, and then makes another trial.
77. So this ability to reset and try again repeatedly is often assumed by pure policy gradient methods, and although it's not technically assumed by most value-based methods, they tend to work best when this assumption is satisfied.
78. It's also assumed by some model-based RL algorithms.
79. Another common assumption, very common in model-based methods especially, is continuity or smoothness.
80. This is assumed by some continuous value function learning methods, and it's often assumed by model-based RL methods derived from optimal control, which really require continuity or smoothness to work well.
81. So as we cover various RL algorithms over the next few weeks, I'll point out some of these assumptions as we go, but keep in mind that many of these methods will differ in the kinds of assumptions they make, and also how rigidly these assumptions must be satisfied in order for those methods to work well in practice.