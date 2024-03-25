1. For the last part of today's lecture, I want to briefly review the material that we've covered, and then discuss some examples of Actor-Critic algorithms in the literature.
2. So this will be a pretty brief section.
3. To summarize what we've covered, we discussed how an Actor-Critic algorithm consists of several parts, an actor, which is the policy, and the critic, which is the value function.
4. The Actor-Critic algorithm can be viewed as a version of policy gradient with substantially reduced variance, and like the policy gradient and all other RL algorithms, it consists of three parts.
5. The orange box where we generate samples, the green box where we estimate our return, which now corresponds to fitting the value function, and a blue box where we use gradient ascent to update our policy, just like in policy gradients.
6. Policy evaluation refers to the process of fitting the value function, and discount factors are something that we can use to make it feasible to do policy evaluation with infinite horizon.
7. It has several different interpretations.
8. You can interpret it as the fear of death, meaning that you would like to receive rewards sooner rather than later before you die, but you can also interpret it as a kind of variance reduction trick.
9. We talked about the design of Actor-Critic algorithms, how you could have one network with two heads or two separate networks, and how you could have batch mode or online Actor-Critic algorithms, and you can use parallelism to get mini-batch sizes that are larger than one.
10. We also talked about state-dependent baselines and even action-dependent control variants as another way to use the critic while remaining unbiased, and we talked about how we can combine these with n-step returns or even the generalized advantage estimator, which averages together many different n-step return estimators.
11. Here are some examples of Actor-Critic algorithms in the literature.
12. This video, which I've showed several times already, is actually from a paper called High-Dimensional Continuous Control with Generalized Advantage Estimators, which introduced the idea of the action-dependent control variance.
13. It introduced the GAE estimator, which is basically a kind of weighted sum of different n-step estimators.
14. This uses a kind of batch mode Actor-Critic, which combines Monte Carlo and function approximator estimators via this GAE trick.
15. And in this paper, the experiments focus on continuous control tasks like this running humanoid robot.
16. Here's another example.
17. This is from a paper called Asynchronous Methods for Deep Reinforcement Learning, and this paper focuses on online Actor-Critic algorithms using parallelized asynchronous systems.
18. So in this particular video, they actually have an image-based Actor-Critic algorithm using a convnet and a recurrent neural network to navigate a maze.
19. They also use n-step returns with n equals 4, and they have a single network for the actor and critic with multiple heads.
20. If you want to read more about Actor-Critic algorithms, here are a few recommendations.
21. Some classic papers.
22. This paper called Policy Grading Methods for Reaction, which is a paper that I wrote about deep reinforcement learning with function approximation.
23. It's actually a very nice paper to read.
24. It describes the theoretical foundations of policy gradients, and also discusses what I referred to as the causality trick before, where you can disregard past rewards at the current time step, and it actually describes how to turn all of this into an Actor-Critic method.
25. So a lot of the material in today's lecture is based on the ideas presented in this paper.
26. Some more recent deep reinforcement learning papers.
27. This is the asynchronous methods paper that I showed on the podcast.
28. This is the asynchronous methods paper that I showed on the podcast.
29. This is the asynchronous methods paper that I showed on the podcast.
30. This is the asynchronous methods paper that I showed on the podcast.
31. And then this is the paper that used action-dependent control variants called QPROP.