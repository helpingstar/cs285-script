1.  In the next part of today's lecture, I'm going to give you a kind of a whirlwind tour through different types of reinforcement learning algorithms.
2. We'll talk about each of these types in much more detail in the next few lectures, but for now we'll just discuss what these types are so they don't come as a surprise later.
3. So the RL algorithms we'll cover will generally be optimizing the RL objective that I defined before.
4. Policy gradient algorithms attempt to directly calculate a derivative of this objective with respect to theta and then perform a gradient descent procedure using that derivative.
5. Value-based methods estimate value functions or Q functions for the optimal policy and then use those value functions or Q functions, which are typically themselves represented by a function approximator like a neural network, to improve the policy.
6. Oftentimes pure value-based functions don't even represent the policy directly but rather represented implicitly as something like an argmax of a Q function.
7. Actrocritic methods are a kind of hybrid between the two.
8. Actrocritic methods learn a Q function or value function and then use it to improve the policy, typically by using them to calculate a better policy gradient.
9. And then model-based reinforcement learning algorithms will also estimate a transition model.
10. They'll estimate some model of the transition probabilities t and then they will either use a transition model for planning directly without any explicit policy or use the transition model to improve the policy.
11. And there are actually many variants in model-based RL for how the transition model can be used.
12. All right, let's start our conversation with model-based RL algorithms.
13. So, let's start our conversation with model-based RL algorithms.
14. So, for model-based RL algorithms, the green box will typically consist of learning some model for p of s given s .
15. So, this could be a neural net that takes in s and either outputs a probability distribution over t plus one or if it's a deterministic model just attempts to predict s directly.
16. And then the blue box has a number of different options.
17. So, let's focus in on that blue box since model-based RL algorithms are usually based on a distribution model.
18. So, let's start our conversation with model-based RL algorithms.
19. So, let's start our conversation with model-based RL algorithms.
20. So, model-based RL algorithms are usually based on a distribution model for planning directly without any explicit policy or So, one option for model-based RL algorithms is to simply use the learned model directly to plan.
21. So, you could, for example, learn how the rules of a chess game work and then use your favorite discrete planning algorithm like Monte Carlo Tree Search to play chess.
22. Or you can learn the physics of a continuous environment for a robot and then use some optimal control over the f of s in terms of how the rules of a chess game work and then use your favorite discrete planning algorithm like Monte Carlo Tree Search to play chess.
23. or trajectory optimization procedure through that learned physics model to control the robot.
24. Another option is to use the learned model to compute derivatives of the reward function with respect to the policy, essentially through backpropagation.
25. This is a very simple idea, but it actually requires quite a few tricks to make it work well, typically in order to account for numerical stability.
26. So, for example, second-order methods tend to work a lot better than first-order methods for backpropagating the policy.
27. Another common use of a model is to use the model to actually learn a separate value function or Q function and then use that value function or Q function to improve the policy.
28. So the value function or Q function would be learned using some type of dynamic programming method.
29. And it's also fairly common to kind of extend number three to essentially use a model to generate additional data for a model-free reinforcement learning algorithm.
30. And that can often work very well.
31. All right, value function-based algorithms.
32. So for value function-based algorithms, the green box involves fitting some estimate of V of S or Q of S comma A, usually using a neural network to represent V of S or Q of S comma A, where the network takes in S or S comma A as input and outputs a real value number.
33. And then the blue box, if it's a pure value-based method, would simply choose a value-based method.
34. And then the blue box would use the policy to be the argmax of QSA.
35. So in a pure value-based method, we wouldn't actually represent the policy explicitly as a neural net.
36. We would just represent it implicitly as an argmax over a neural net representing QSA.
37. Direct policy gradient methods would implement the blue box simply by taking a gradient step, a gradient ascent step on theta using the gradient of the expected value of the reward.
38. We'll talk about how this gradient can be estimated in the next lecture.
39. But the green box for policy gradient algorithms is very, very simple.
40. It just involves computing the total reward along each trajectory simply by adding up the rewards that were obtained during the rollout.
41. By the way, when I use the term rollout, that simply means sample of your policy.
42. It means run your policy one step at a time.
43. And the reason we call it a rollout is because you're unrolling your policy one step at a time.
44. Actor-critic algorithms.
45. Actor-critic algorithms are a very common term for policy gradients.
46. They're often used to calculate the total reward of a policy.
47. They're a kind of hybrid between value-based methods and policy gradient methods.
48. Actor-critic algorithms also fit a value function or a Q function in the green box, just like value-based methods.
49. But then in the blue box, they actually take a gradient ascent step on the policy, just like policy gradient methods, utilizing the value function or Q function to obtain a better estimate of the gradient, a more accurate gradient.