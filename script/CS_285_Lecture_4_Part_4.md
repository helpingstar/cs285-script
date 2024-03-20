1. In the next part of today's lecture, I'm going to give you a kind of a whirlwind tour through different types of reinforcement learning algorithms.
2. We'll talk about each of these types in much more detail in the next few lectures, but for now we'll just discuss what these types are so they don't come as a surprise later.
3. So the RL algorithms we'll cover will generally be optimizing the RL objective that I defined before.
4. Policy gradient algorithms attempt to directly calculate a derivative of this objective with respect to θ and then perform a gradient descent procedure using that derivative.
5. Value-based methods estimate value functions or Q functions for the optimal policy and then use those value functions or Q functions, which are typically themselves represented by a function approximator like a neural network, to improve the policy.
6. Oftentimes pure value-based functions don't even represent the policy directly but rather represented implicitly as something like an argmax of a Q function.
7. Actor-critic methods are a kind of hybrid between the two.
8. Actor-critic methods learn a Q function or value function and then use it to improve the policy, typically by using them to calculate a better policy gradient.
9. And then model-based reinforcement learning algorithms will estimate a transition model.
10. They'll estimate some model of the transition probabilities t and then they will either use a transition model for planning directly without any explicit policy or use the transition model to improve the policy.
11. And there are actually many variants in model-based RL for how the transition model can be used.
12. All right, let's start our conversation with model-based RL algorithms.
13. So, for model-based RL algorithms, the green box will typically consist of learning some model for p(s_{t+1} | s_t, a_t).
14. So, this could be a neural net that takes in s_t, a_t and either outputs a probability distribution over t plus one or if it's a deterministic model just attempts to predict s_{t+1} directly.
15. And then the blue box has a number of different options.
16. So, let's focus in on that blue box since model-based RL algorithms will differ greatly in terms of how they implement this part.
17. So one option for model based RL algorithms is to simply use the learned model directly to plan.
18. So, you could, for example, learn how the rules of a chess game work and then use your favorite discrete planning algorithm like Monte Carlo Tree Search to play chess.
19. Or you can learn the physics of a continuous environment for a robot and then use some optimal control over the trajectory optimization procedure through that learned physics model to control the robot.
20. Another option is to use the learned model to compute derivatives of the reward function with respect to the policy, essentially through backpropagation.
21. This is a very simple idea, but it actually requires quite a few tricks to make it work well, typically in order to account for numerical stability.
22. So, for example, second-order methods tend to work a lot better than first-order methods for backpropagating the policy.
23. Another common use of a model is to use the model to actually learn a separate value function or Q function and then use that value function or Q function to improve the policy.
24. So the value function or Q function would be learned using some type of dynamic programming method.
25. And it's also fairly common to kind of extend number three to essentially use a model to generate additional data for a model-free reinforcement learning algorithm.
26. And that can often work very well.
27. All right, value function-based algorithms.
28. So for value function-based algorithms, the green box involves fitting some estimate of V(s) or Q(s,a), usually using a neural network to represent V(s) or Q(s,a), where the network takes in s or (s,a) as input and outputs a real valued number.
29. And then the blue box, if it's a pure value-based method, would simply choose the policy to be the argmax of Q(s,a).
30. So in a pure value-based method, we wouldn't actually represent the policy explicitly as a neural net.
31. We would just represent it implicitly as an argmax over a neural net representing Q(s,a).
32. Direct policy gradient methods would implement the blue box simply by taking a gradient step, a gradient ascent step on θ using the gradient of the expected value of the reward.
33. We'll talk about how this gradient can be estimated in the next lecture.
34. But the green box for policy gradient algorithms is very, very simple.
35. It just involves computing the total reward along each trajectory simply by adding up the rewards that were obtained during the rollout.
36. By the way, when I use the term rollout, that simply means sample of your policy.
37. It means run your policy one step at a time.
38. And the reason we call it a rollout is because you're unrolling your policy one step at a time.
39. Actor-critic algorithms are a kind of hybrid between value-based methods and policy gradient methods.
40. Actor-critic algorithms also fit a value function or a Q function in the green box, just like value-based methods.
41. But then in the blue box, they actually take a gradient ascent step on the policy, just like policy gradient methods, utilizing the value function or Q function to obtain a better estimate of the gradient, a more accurate gradient.