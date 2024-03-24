1. In the next portion of today's lecture, we're going to talk about implementing policy gradients in practice, in deep RL algorithms.
2. One of the main challenges with implementing policy gradients is that we would like to implement them in such a way that automatic differentiation tools like TensorFlow or PyTorch can calculate the policy gradient for us with reasonable computational and memory requirements.
3. If we wanted to implement policy gradients naively, we could simply calculate ∇log π (a_{i,t}|s_{i,t}) for every single state action tuple that we sampled.
4. However, typically, this is very inefficient because neural networks can have a very large number of parameters.
5. In fact, the number of parameters is usually much larger than the number of samples that we've produced.
6. So let's say that we have N parameters, where N might be on the order of a million.
7. And we have...
8. 100 trajectories, each with 100 time steps.
9. So we have 10,000 total state action pairs, which means that we're going to need to calculate 10,000 of these 1 million length vectors.
10. That's going to be very, very expensive in terms of memory storage and also computationally.
11. Typically, when we want to calculate derivatives through neural networks efficiently, we want to utilize the back propagation algorithm.
12. So instead of calculating the derivative of the neural net's output with respect to its input, and then multiplying that by the derivative of the loss, we do the opposite.
13. We first calculate the derivative of the loss, and then back propagate it through the neural network using the back propagation algorithm, which is what our automatic differentiation tools will do for us.
14. In order to do that, we need to set up a graph such that the derivative of that graph gives us the policy gradient.
15. All right.
16. So how do we compute policy gradients with automatic differentiation?
17. Well, we need a graph such that its gradient is the policy gradient.
18. The way that we're going to figure this out is by starting with the gradients that we already know how to compute, which are maximum likelihood gradients.
19. So if we want to compute maximum likelihood gradients, then what we would do is we would implement the maximum likelihood objective using something like a cross-entropy loss, and then call dot backward or dot gradients on it, depending on your automatic differentiation package, and obtain your gradients.
20. So the way that we're going to implement policy gradients to get our auditive package to calculate them efficiently, is by implementing a kind of pseudo loss as a weighted maximum likelihood.
21. So instead of implementing J maximum likelihood, we'll implement this thing called tilde{J}, which will just be the sum of the log probabilities of all of our sampled actions, multiplied by the rewards to go, ^{Q}.
22. Now critically, this equation is not the reinforcement learning objective.
23. In fact, this equation is not anything.
24. It's just a quantity, chosen such that its derivatives come out to be the policy gradient.
25. Of course, a critical portion of this is that our automatic differentiation package doesn't realize that those ^{Q} numbers are themselves affected by our policy.
26. So it's just dealing with the graph that we provided it.
27. So in a sense, we're almost trying to trick our auditive package into giving us the gradient that we want.
28. Okay, so here log π is, you know, would be for example our cross-entropy loss.
29. If we have discrete actions or squared error if we have normally distributed continuous actions.
30. All right, so I have some pseudocode here.
31. This pseudocode is actually in TensorFlow because I taught the class in TensorFlow in past years.
32. You're going to be doing the the policy gradients assignment in PyTorch.
33. The basic idea is very much the same.
34. It's just the particular terminology is going to be a little different.
35. But hopefully the pseudocode is still straightforward for everyone to parse.
36. So the pseudocode that I have here is the pseudocode for maximum likelihood learning.
37. This is supervised learning.
38. Here, actions is a tensor with dimensionality (N*T) along the first dimension.
39. So number of samples times the number of time steps and the dimensionality of the action along the second dimension.
40. And states is a tensor (N*T) × the number of state dimensions.
41. So the first line logits equals policy dot predictions states that simply asks the policy network to make predictions for those states.
42. Basically output the logits over the actions.
43. This is a discrete action example.
44. Then the second line negative likelihoods basically uses the softmax cross entropy function to produce likelihoods for all the actions.
45. And then we do a mean reduce on those and calculate their gradients.
46. So this will give you the gradient of the likelihood.
47. This is what you do for supervised learning.
48. To implement policy gradients, you just have to put in weights to get a weighted likelihood and those weights correspond to those reward-to-go values.
49. So I'm going to assume that the reward-to-go values are all packed into a tensor called `q_values`, which is an n times t by 1 tensor.
50. And then after I calculate my likelihoods, I'll turn them into weighted likelihoods by pointwise multiplying them by the `q_values`.
51. And that's the only change that I make.
52. Then I mean reduce those and then I call their gradients.
53. So this will essentially trick your auditive package into calculating a policy gradient.
54. So in math, what we've implemented is this.
55. We've basically turned our maximum likelihood loss into this modified pseudo loss tilde{J} where we weight our likelihoods by hat{Q}s.
56. And of course it's up to you to actually implement some code to compute those Q values, which you could do simply in NumPy.
57. You don't really need to use your auditive package to compute those.
58. All right a few general tips about using policy gradients in practice.
59. First, remember that the policy gradient has high variance.
60. So even though the implementation looks a lot like supervised learning, it's going to behave very differently from supervised learning.
61. The high variance of the policy gradient will make some things quite a bit harder.
62. It means your gradients will be very noisy.
63. Which means that you potentially probably need to use larger batches, probably much larger than what you're used to for supervised learning.
64. So batch sizes in the thousands or tens of thousands are fairly typical.
65. Tweaking the learning rate is going to be substantially harder.
66. Adaptive step size rules like ADAM can be okay-ish, but just regular SGD with momentum can be extremely hard to use.
67. We'll learn about policy gradient-specific learning rate adjustment methods later when we talk about things like natural gradient, but for now, using ADAM is a good starting point.
68. And in general, just expect to have to do more hyperparameter tuning than you've usually had to do for supervised learning.
69. So just to review, we talked about how the policy gradient is on policy, how we can derive an off policy variant using importance sampling, which unfortunately has exponential scaling in the time horizon, but we can ignore the state portion, which gives us an approximation.
70. We talked about how we can implement policy gradients with automatic differentiation, and the key to doing that is setting it up so that AutoDiff back-propagates things for us properly by using the pseudo-laws.
71. And we talked about some practical considerations, batch size, learning rates, and optimizers.
