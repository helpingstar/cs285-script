1. All right, next let's get started discussing some reinforcement learning algorithms.
2. There are quite a few reinforcement learning algorithms that we'll cover in this course, but at a high level, all of these algorithms will share more or less the same high-level anatomy.
3. They will consist of three basic parts.
4. The first part, which I will always draw in orange, is to generate samples.
5. So reinforcement learning is about learning through trial and error in some sense.
6. And the trial part of trial and error means actually attempting to run your policy in your environment, actually have it interact with your Markov decision process, and collect samples.
7. What are samples?
8. Well, samples are trajectories.
9. So you will interact with your MDP, and typically what you will be doing is you will be sampling trajectories from the trajectory distribution, typically the one induced by your policy, although when we talk about exploration later, sometimes you might actually choose to sample from a slightly different trajectory distribution than the one that our policy would define.
10. But for now, let's just assume that generating samples means sampling trajectories from the trajectory distribution defined by your policy, which simply means running your policy in the environment.
11. Then we will have the green box, and the green box corresponds to learning some kind of model.
12. This could be literally a model of the dynamics in a model-based RL algorithm, or it can be some sort of more implicit model such as a value function.
13. And this green box basically corresponds to estimating something about your current policy, something about how your policy is doing, how well it's performing, what kind of rewards it's attaining.
14. And then we'll have the blue box, which is where you actually change your policy to make it better.
15. And then you repeat this process.
16. Pretty much all of the algorithms that we will cover will have these three parts.
17. In some cases, one of these parts might be very, very simple.
18. In some cases, some of them might be very complex.
19. But all three of them are generally going to be present.
20. So here are some simple examples.
21. Let's say that we're going to run our policy and we'll generate sample trajectories denoted by these black lines and we'll evaluate them to see if they're good or bad.
22. So evaluate means just sum up their rewards.
23. Summing up their rewards is what happens in the green box.
24. And then when we improve the policy, we might try to make the good trajectories, kind of the green check marks, be more likely, and the bad trajectories, the red ones, be less likely.
25. So that's the improvement step.
26. Now what I'm describing here is the basic high-level scheme for a policy gradient algorithm, which we'll learn about in detail in the next lecture.
27. So in a policy gradient algorithm, the green box, the box where we estimate something about our policy, is very, very simple.
28. It simply consists of summing up the rewards along the trajectories that we sampled, and that tells us how good our policy is.
29. So the green box is just a summation.
30. The blue box might involve calculating the gradient of the reward of your policy, and we'll talk about how to do that in the next lecture, and applying that gradient to the policy parameters, θ.
31. So that's a very simple trial and error style reinforcement learning algorithm.
32. Run your policy, get some trajectories, measure how good those trajectories are, and then modify the policy to make the better trajectories have a higher probability.
33. You could also imagine doing a model-based RL procedure.
34. You can kind of think of this as RL by back-prop.
35. So maybe in the green box, you learn a model, you learn some other neural network, f_ϕ, such that s_{t+1}, is approximately equal to f_ϕ(s_t, a_t) , and you train f_ϕ on data generated in the orange box with supervised learning.
36. So maybe you have a whole other neural network that goes from s_t, a_t to s_{t+1} .
37. Now this green box is much more complex than the one we had on the previous slide.
38. On the previous slide, we would just sum over the rewards in our trajectories.
39. Here, we're actually fitting a whole other neural network.
40. The summation might take a millisecond.
41. This might take minutes to train, or maybe even hours if it's using images.
42. And then in the blue box, we might back-prop through f and r to train the policy π_θ.
43. So if we want to calculate the reward of the policy, we can basically compose the policy with f, actually use our automatic differentiation software to calculate the reward, and back-prop through all of them to optimize the policy.
44. We're going to cover methods that do some variant of this when we talk about model-based reinforcement learning much later in the course.
45. If the details of this don't currently make sense, don't worry about it.
46. The only point of this slide is to explain to you the different incarnations of the green box and the blue box that we might see in some very different reinforcement learning algorithms.
47. All right.
48. Now, which parts in this whole process are expensive, and which parts might be cheap?
49. Well, the orange box, its cost in terms of time and computation depends a great deal on what kind of problem you're solving.
50. If you're collecting samples by running a real-world system like a real robot, a real car, a real power grid, a real chemical plant, whatever, the orange box can be potentially extremely expensive, right?
51. Because you have to collect data in real time, at least until we invent time travel.
52. And if you need, like, thousands of samples each iteration of your RL algorithm, this can be very, very costly.
53. On the other hand, if you're collecting samples in the MuJoCo simulator that all of you are using for homework one, then, you know, the MuJoCo simulator can run at up to 10,000x real time, so the cost of the orange box might actually be trivial.
54. So depending on which of these regimes you're in or where you are on the spectrum, you might care more or less about how many samples you need in the orange box, which will influence your choice of reinforcement learning algorithms.
55. So this can range from prohibitively expensive to trivially cheap, depending on how you're learning.
56. The green box also could range from extremely cheap to extremely expensive.
57. So if you're just estimating the return of your policy by summing up the rewards that you obtained, this is very, very cheap.
58. It's just a summation operator.
59. If you are learning an entire model by training a whole other neural net, this might be very expensive.
60. It might require, you know, a big supervised learning run in the inner loop of your RL algorithm.
61. Similarly, in the blue box, if you're just taking one gradient step, this might be fairly cheap.
62. If you have to backprop through your model and your policy like I discussed in the model-based slide.
63. This might be very expensive.
64. And there will be algorithms that fall at different points of the spectrum for different boxes.
65. For instance, a Q-learning algorithm, which we'll cover a couple weeks from now, basically spends all of its effort in the green box, and the blue box is just an argmax.